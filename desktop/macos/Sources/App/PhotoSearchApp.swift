import SwiftUI

/// Entry point: `--self-test …` runs the headless end-to-end check (see SelfTest.swift) and exits; otherwise the app.
@main
enum Main {
    static func main() {
        let args = Array(CommandLine.arguments.dropFirst())
        if args.first == "--self-test" {
            let done = DispatchSemaphore(value: 0)
            nonisolated(unsafe) var code: Int32 = 1
            Task.detached {
                code = await SelfTest.run(Array(args.dropFirst()))
                done.signal()
            }
            done.wait()
            exit(code)
        }
        PhotoSearchApp.main()
    }
}

struct PhotoSearchApp: App {
    @State private var model = AppModel()

    var body: some Scene {
        WindowGroup("Photo Search") {
            ContentView(model: model)
                .frame(minWidth: 760, minHeight: 480)
        }
        .commands {
            CommandGroup(after: .newItem) {
                Button("Add Folder…") { model.chooseFolder() }.keyboardShortcut("o")
            }
        }
    }
}
