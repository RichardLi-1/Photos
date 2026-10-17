import PhotosUI
import SwiftUI

/// Floating glass search capsule at the bottom (web/src/components/SearchBar.tsx): type to search; the camera button
/// searches by a photo picked from the system picker (out of process, no library permission) or taken with the camera.
struct SearchCapsule: View {
    @Bindable var model: AppModel
    @FocusState private var focused: Bool
    @State private var pickerItem: PhotosPickerItem?
    @State private var showPicker = false
    @State private var showCamera = false

    private var queryImage: UIImage? {
        if case .image(let image, _) = model.query { return image }
        return nil
    }

    var body: some View {
        HStack(spacing: 6) {
            if let queryImage {
                Image(uiImage: queryImage)
                    .resizable().scaledToFill()
                    .frame(width: 36, height: 36)
                    .clipShape(RoundedRectangle(cornerRadius: 9, style: .continuous))
                    .padding(.leading, 6)
                    .accessibilityLabel("Searching by your photo")
            } else {
                Image(systemName: "magnifyingglass")
                    .foregroundStyle(.secondary)
                    .padding(.leading, 12)
                    .accessibilityHidden(true)
            }
            TextField(placeholder, text: $model.searchText)
                .font(.title3)
                .focused($focused)
                .submitLabel(.search)
                .onSubmit { model.submit() }
                .autocorrectionDisabled()
                .frame(minHeight: 44)
                .accessibilityLabel("Search photos")
            if !model.searchText.isEmpty || model.query != nil {
                Button {
                    model.clear()
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(.title3)
                        .foregroundStyle(.secondary)
                        .frame(minWidth: 44, minHeight: 44)
                        .contentShape(Rectangle())
                }
                .accessibilityLabel("Clear search")
            }
            Menu {
                Button { showPicker = true } label: { Label("Choose Photo", systemImage: "photo.on.rectangle") }
                if CameraPicker.isAvailable {
                    Button { showCamera = true } label: { Label("Take Photo", systemImage: "camera") }
                }
            } label: {
                Image(systemName: "camera.viewfinder")
                    .font(.title2)
                    .frame(minWidth: 44, minHeight: 44)
                    .contentShape(Rectangle())
            }
            .accessibilityLabel("Search with a photo")
            .padding(.trailing, 6)
        }
        .padding(.vertical, 6)
        .glassBackground(in: Capsule())
        .photosPicker(isPresented: $showPicker, selection: $pickerItem, matching: .images, preferredItemEncoding: .compatible)
        .onChange(of: pickerItem) { _, item in
            guard let item else { return }
            Task {
                if let data = try? await item.loadTransferable(type: Data.self), let image = UIImage(data: data) {
                    model.search(image: image)
                }
                pickerItem = nil
            }
        }
        .fullScreenCover(isPresented: $showCamera) {
            CameraPicker { image in
                showCamera = false
                if let image { model.search(image: image) }
            }
            .ignoresSafeArea()
        }
    }

    private var placeholder: LocalizedStringKey {
        if queryImage != nil { return "Add words to refine" }
        if model.progress.embedded > 0 { return "Search \(model.progress.embedded) photos" }
        return "Search photos"
    }
}

/// Status pills above the capsule: search status and indexing progress (tap → Settings).
struct StatusPills: View {
    let model: AppModel
    var onShowSettings: () -> Void

    var body: some View {
        VStack(spacing: 8) {
            if let status = model.status {
                HStack(spacing: 8) {
                    if model.isSearching { ProgressView().controlSize(.small) }
                    Text(status)
                }
                .font(.footnote)
                .padding(.horizontal, 14).padding(.vertical, 8)
                .glassBackground(in: Capsule())
                .accessibilityElement(children: .combine)
                .accessibilityAddTraits(.updatesFrequently)
            }
            if let line = indexingLine {
                Button(action: onShowSettings) {
                    HStack(spacing: 8) {
                        ProgressView(value: model.progress.fraction)
                            .frame(width: 56)
                            .accessibilityHidden(true)
                        Text(line)
                    }
                    .font(.footnote)
                    .padding(.horizontal, 14)
                    .frame(minHeight: 44)
                    .glassBackground(in: Capsule())
                }
                .buttonStyle(.plain)
                .accessibilityHint("Opens indexing settings")
            }
        }
    }

    private var indexingLine: LocalizedStringResource? {
        let p = model.progress
        let done = p.embedded + p.skipped
        switch p.state {
        case .preparing: return "Preparing your library…"
        case .running where done < p.total: return "Indexing \(done) of \(p.total)"
        case .paused(let reason) where done < p.total: return Self.text(for: reason)
        default: return nil
        }
    }

    static func text(for reason: PauseReason) -> LocalizedStringResource {
        switch reason {
        case .thermal: "Indexing paused while iPhone cools down"
        case .lowPower: "Indexing paused in Low Power Mode"
        case .lowBattery: "Indexing paused: low battery"
        case .lowDisk: "Indexing paused: storage almost full"
        case .userPaused: "Indexing paused"
        }
    }
}
