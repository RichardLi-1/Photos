import Foundation
import SQLite3

/// Minimal wrapper over the system sqlite3 C API. Not thread-safe by itself: owned by one actor (`LibraryStore`).
final class SQLite {
    enum Value { case null, int(Int64), real(Double), text(String), blob(Data) }

    struct Error: Swift.Error, CustomStringConvertible {
        let code: Int32, message: String
        var description: String { "SQLite \(code): \(message)" }
    }

    private var db: OpaquePointer?
    private static let transient = unsafeBitCast(-1, to: sqlite3_destructor_type.self)

    init(path: String) throws {
        let flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_NOMUTEX | SQLITE_OPEN_FILEPROTECTION_COMPLETEUNTILFIRSTUSERAUTHENTICATION
        guard sqlite3_open_v2(path, &db, flags, nil) == SQLITE_OK else {
            defer { sqlite3_close(db) }
            throw Error(code: sqlite3_errcode(db), message: String(cString: sqlite3_errmsg(db)))
        }
        sqlite3_busy_timeout(db, 5_000)
    }

    deinit { sqlite3_close_v2(db) }

    var lastInsertID: Int64 { sqlite3_last_insert_rowid(db) }
    var changes: Int { Int(sqlite3_changes(db)) }

    func execute(_ sql: String) throws {
        var err: UnsafeMutablePointer<CChar>?
        guard sqlite3_exec(db, sql, nil, nil, &err) == SQLITE_OK else {
            let message = err.map { String(cString: $0) } ?? "?"
            sqlite3_free(err)
            throw Error(code: sqlite3_errcode(db), message: message)
        }
    }

    /// Runs `sql` with `args`; returns all rows (each a column-name → value dictionary is overkill; positional arrays).
    @discardableResult
    func query(_ sql: String, _ args: [Value] = []) throws -> [[Value]] {
        var rows: [[Value]] = []
        try each(sql, args) { rows.append($0) }
        return rows
    }

    func run(_ sql: String, _ args: [Value] = []) throws {
        try each(sql, args) { _ in }
    }

    func each(_ sql: String, _ args: [Value] = [], _ row: ([Value]) throws -> Void) throws {
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { throw lastError() }
        defer { sqlite3_finalize(stmt) }
        for (i, arg) in args.enumerated() {
            let idx = Int32(i + 1)
            switch arg {
            case .null: sqlite3_bind_null(stmt, idx)
            case .int(let v): sqlite3_bind_int64(stmt, idx, v)
            case .real(let v): sqlite3_bind_double(stmt, idx, v)
            case .text(let v): sqlite3_bind_text(stmt, idx, v, -1, Self.transient)
            case .blob(let v): _ = v.withUnsafeBytes { sqlite3_bind_blob(stmt, idx, $0.baseAddress, Int32(v.count), Self.transient) }
            }
        }
        while true {
            let rc = sqlite3_step(stmt)
            if rc == SQLITE_DONE { return }
            guard rc == SQLITE_ROW else { throw lastError() }
            let n = sqlite3_column_count(stmt)
            try row((0..<n).map { column(stmt, $0) })
        }
    }

    func transaction<T>(_ body: () throws -> T) throws -> T {
        try execute("BEGIN IMMEDIATE")
        do {
            let out = try body()
            try execute("COMMIT")
            return out
        } catch {
            try? execute("ROLLBACK")
            throw error
        }
    }

    private func column(_ stmt: OpaquePointer?, _ i: Int32) -> Value {
        switch sqlite3_column_type(stmt, i) {
        case SQLITE_INTEGER: return .int(sqlite3_column_int64(stmt, i))
        case SQLITE_FLOAT: return .real(sqlite3_column_double(stmt, i))
        case SQLITE_TEXT: return .text(String(cString: sqlite3_column_text(stmt, i)))
        case SQLITE_BLOB:
            let n = Int(sqlite3_column_bytes(stmt, i))
            guard n > 0, let p = sqlite3_column_blob(stmt, i) else { return .blob(Data()) }
            return .blob(Data(bytes: p, count: n))
        default: return .null
        }
    }

    private func lastError() -> Error {
        Error(code: sqlite3_errcode(db), message: String(cString: sqlite3_errmsg(db)))
    }
}

extension SQLite.Value {
    var int: Int64? { if case .int(let v) = self { v } else { nil } }
    var double: Double? {
        switch self {
        case .real(let v): v
        case .int(let v): Double(v)
        default: nil
        }
    }
    var string: String? { if case .text(let v) = self { v } else { nil } }
    var data: Data? { if case .blob(let v) = self { v } else { nil } }

    init(_ v: Double?) { self = v.map { .real($0) } ?? .null }
    init(_ v: String?) { self = v.map { .text($0) } ?? .null }
}
