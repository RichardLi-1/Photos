import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @Bindable var model: AppModel
    @State private var dropTargeted = false

    var body: some View {
        NavigationSplitView {
            Sidebar(model: model)
                .navigationSplitViewColumnWidth(min: 210, ideal: 240, max: 320)
        } detail: {
            ResultsView(model: model)
                .overlay {
                    if dropTargeted {
                        RoundedRectangle(cornerRadius: 12).strokeBorder(Color.accentColor, lineWidth: 3).padding(6)
                            .allowsHitTesting(false)
                    }
                }
                .dropDestination(for: URL.self) { urls, _ in
                    model.handleDrop(urls)
                    return !urls.isEmpty
                } isTargeted: { dropTargeted = $0 }
                .inspector(isPresented: Binding(get: { model.selection != nil }, set: { if !$0 { model.selection = nil } })) {
                    if let hit = model.hit(model.selection) {
                        MatchCard(model: model, hit: hit).inspectorColumnWidth(min: 280, ideal: 340, max: 480)
                    }
                }
        }
        .searchable(text: $model.queryText, placement: .toolbar, prompt: Text("Describe a photo: “dog on the beach”, “menu with prices”"))
        .onSubmit(of: .search) { model.submitSearch() }
        .onChange(of: model.queryText) { _, new in if new.isEmpty, model.mode != .browse { model.showAll() } }
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button { model.chooseFolder() } label: { Label("Add Folder", systemImage: "folder.badge.plus") }
                    .help("Add a folder of photos to search")
            }
        }
        .alert("Photo Search", isPresented: Binding(get: { model.errorMessage != nil }, set: { if !$0 { model.errorMessage = nil } })) {
            Button("OK", role: .cancel) {}
        } message: { Text(model.errorMessage ?? "") }
        .task { await model.start() }
    }
}

struct Sidebar: View {
    @Bindable var model: AppModel

    var body: some View {
        List {
            Section("Library") {
                Button { model.queryText = ""; model.showAll() } label: { Label("All Photos", systemImage: "photo.on.rectangle") }
                    .buttonStyle(.plain)
            }
            Section("Folders") {
                ForEach(model.folders) { f in
                    Label {
                        Text(f.url.lastPathComponent).lineLimit(1).truncationMode(.middle)
                            .foregroundStyle(f.available ? .primary : .secondary)
                    } icon: { Image(systemName: f.available ? "folder" : "externaldrive.badge.xmark") }
                    .help(f.available ? f.url.path : String(localized: "Not available right now: \(f.url.path)"))
                    .contextMenu {
                        Button("Show in Finder") { NSWorkspace.shared.activateFileViewerSelecting([f.url]) }
                        Button("Remove from Photo Search", role: .destructive) { model.removeFolder(f) }
                    }
                }
                Button { model.chooseFolder() } label: { Label("Add Folder…", systemImage: "plus") }.buttonStyle(.plain)
                    .foregroundStyle(.secondary)
            }
            Section("Photos") {
                Toggle(isOn: Binding(get: { model.includePhotos }, set: { on in Task { await model.setIncludePhotos(on) } })) {
                    Label("Photos Library", systemImage: "photo.stack")
                }
                .toggleStyle(.switch).controlSize(.small)
                .help("Also index your Photos library. Only photos stored on this Mac are read; iCloud originals are never downloaded.")
            }
        }
        .listStyle(.sidebar)
        .safeAreaInset(edge: .bottom) { StatusPanel(model: model).padding(10) }
    }
}

struct StatusPanel: View {
    let model: AppModel

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            if let err = model.modelError {
                Label(err, systemImage: "exclamationmark.triangle").font(.caption).foregroundStyle(.red)
            } else if !model.modelsReady {
                HStack(spacing: 6) { ProgressView().controlSize(.small); Text("Loading on-device models…").font(.caption) }
            } else {
                let s = model.status
                HStack {
                    Text(statusText(s)).font(.caption).foregroundStyle(.secondary).lineLimit(2)
                    Spacer()
                    if s.isBusy {
                        Button { Task { await model.setPaused(true) } } label: { Image(systemName: "pause.fill") }
                            .buttonStyle(.borderless).help("Pause indexing")
                    } else if case .paused = s.state {
                        Button { Task { await model.setPaused(false) } } label: { Image(systemName: "play.fill") }
                            .buttonStyle(.borderless).help("Resume indexing")
                    }
                }
                if s.isBusy || s.state != .finished && s.total > 0 && s.fraction < 1 {
                    ProgressView(value: s.fraction).controlSize(.small)
                }
            }
            Label("On this Mac only · no network access", systemImage: "lock.shield")
                .font(.caption2).foregroundStyle(.tertiary)
                .help("Photos, the index and your searches never leave this Mac. The app is sandboxed without network permission.")
        }
        .padding(10)
        .glassBackground(in: RoundedRectangle(cornerRadius: 12))
    }

    private func statusText(_ s: IndexStatus) -> String {
        let n = s.embedded.formatted()
        switch s.state {
        case .scanning: return String(localized: "Looking for photos…")
        case .running: return String(localized: "Indexing \(n) of \(s.total.formatted())")
        case .paused(let why): return String(localized: "\(why) · \(n) of \(s.total.formatted()) indexed")
        case .finished, .idle:
            if s.total == 0 { return String(localized: "Add a folder to start") }
            return s.skipped > 0 ? String(localized: "\(n) photos indexed · \(s.skipped) unreadable") : String(localized: "\(n) photos indexed")
        }
    }
}

struct ResultsView: View {
    let model: AppModel
    private let columns = [GridItem(.adaptive(minimum: 150, maximum: 260), spacing: 4)]

    var body: some View {
        Group {
            if model.hits.isEmpty {
                EmptyState(model: model)
            } else {
                ScrollView {
                    LazyVGrid(columns: columns, spacing: 4) {
                        ForEach(model.hits) { hit in
                            Cell(model: model, hit: hit)
                        }
                    }
                    .padding(8)
                }
            }
        }
        .safeAreaInset(edge: .top) {
            if model.mode != .browse { ResultHeader(model: model).padding(.horizontal, 12).padding(.top, 8) }
        }
        .navigationTitle(title)
    }

    private var title: String {
        switch model.mode {
        case .browse: String(localized: "All Photos")
        case .text(let q): q
        case .similar: String(localized: "More Like This")
        case .samePlace: String(localized: "Same Place")
        case .photo(let name): name
        }
    }
}

struct Cell: View {
    let model: AppModel
    let hit: SearchHit

    var body: some View {
        let selected = model.selection == hit.id
        Color.clear
            .aspectRatio(1, contentMode: .fit)
            .overlay { Thumbnail(localID: hit.localID) }
            .clipShape(RoundedRectangle(cornerRadius: 6))
            .overlay { RoundedRectangle(cornerRadius: 6).strokeBorder(Color.accentColor, lineWidth: selected ? 3 : 0) }
            .contentShape(Rectangle())
            .onTapGesture(count: 2) { model.open(hit) }
            .onTapGesture { model.selection = selected ? nil : hit.id }
            .contextMenu {
                Button("More Like This") { model.moreLikeThis(hit.id) }
                if model.placeAvailable { Button("Same Place") { model.samePlace(hit.id) } }
                Divider()
                Button("Open") { model.open(hit) }
                if PhotoRef(localID: hit.localID)?.fileURL != nil { Button("Show in Finder") { model.revealInFinder(hit) } }
            }
            .accessibilityElement()
            .accessibilityLabel(PhotoRef(localID: hit.localID)?.displayName ?? "")
            .accessibilityAddTraits(selected ? [.isImage, .isSelected] : .isImage)
            .accessibilityAction(named: "More Like This") { model.moreLikeThis(hit.id) }
    }
}

struct ResultHeader: View {
    let model: AppModel

    var body: some View {
        HStack(spacing: 10) {
            Button { model.queryText = ""; model.showAll() } label: { Image(systemName: "chevron.backward") }
                .buttonStyle(.borderless).help("Back to all photos")
            VStack(alignment: .leading, spacing: 2) {
                Text(headline).font(.callout.weight(.semibold)).lineLimit(1)
                if let u = model.understood { Text("Understood: \(u)").font(.caption).foregroundStyle(.secondary) }
            }
            Spacer()
            if model.searching { ProgressView().controlSize(.small) }
            else if let e = model.elapsed {
                Text("\(model.hits.count) results · \(e.formatted(.units(allowed: [.milliseconds], width: .narrow)))")
                    .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
            }
        }
        .padding(.horizontal, 12).padding(.vertical, 8)
        .glassBackground(in: Capsule())
    }

    private var headline: String {
        switch model.mode {
        case .browse: ""
        case .text(let q): String(localized: "Photos matching “\(q)”")
        case .similar: String(localized: "More like this photo")
        case .samePlace: String(localized: "Same room or place (our trained head)")
        case .photo(let name): String(localized: "Photos like “\(name)”")
        }
    }
}

struct EmptyState: View {
    let model: AppModel

    var body: some View {
        VStack(spacing: 14) {
            Image(systemName: model.folders.isEmpty && !model.includePhotos ? "photo.on.rectangle.angled" : "magnifyingglass")
                .font(.system(size: 44)).foregroundStyle(.secondary)
            if model.folders.isEmpty && !model.includePhotos {
                Text("Search your photos by describing them").font(.title2.weight(.semibold))
                Text("Add a folder, or drop one here. Photo Search reads your photos on this Mac, understands them with on-device models, and never sends anything anywhere.")
                    .multilineTextAlignment(.center).foregroundStyle(.secondary).frame(maxWidth: 440)
                Button("Add Folder…") { model.chooseFolder() }.controlSize(.large).keyboardShortcut("o")
            } else if model.mode == .browse {
                Text(model.status.isBusy ? "Indexing your photos…" : "No photos indexed yet").font(.title3)
            } else if !model.searching {
                Text("No matches").font(.title3)
                Text("Try different words, or fewer filters.").foregroundStyle(.secondary)
            }
        }
        .padding(40)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
