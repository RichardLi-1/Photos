import Photos
import SwiftUI

/// Indexing status, privacy switches and library access, in a standard grouped Form.
struct SettingsView: View {
    @Bindable var model: AppModel
    @AppStorage(PlaceResolver.settingKey) private var placeLookup = false
    @State private var paused = false
    @Environment(\.dismiss) private var dismiss
    @Environment(\.openURL) private var openURL

    var body: some View {
        NavigationStack {
            Form {
                Section("Index") {
                    LabeledContent("Photos indexed", value: "\(model.progress.embedded) of \(model.progress.total)")
                    if model.progress.skipped > 0 {
                        LabeledContent("Not available on this iPhone", value: "\(model.progress.skipped)")
                    }
                    if case .paused(let reason) = model.progress.state, reason != .userPaused {
                        Text(StatusPills.text(for: reason)).foregroundStyle(.secondary)
                    }
                    Toggle("Pause Indexing", isOn: $paused)
                        .onChange(of: paused) { _, value in Task { await model.setIndexingPaused(value) } }
                    if let error = model.modelError {
                        Text("Models unavailable: \(error)").font(.footnote).foregroundStyle(.red)
                    }
                }
                Section {
                    Toggle("Look Up Place Names", isOn: $placeLookup)
                        .onChange(of: placeLookup) { _, on in if on { Task { await model.start() } } }
                } header: {
                    Text("Places")
                } footer: {
                    Text("Off by default. When on, the locations of your photos are sent to Apple to find place names, so you can search for “Paris” or “Lisbon”. Nothing else ever leaves this iPhone.")
                }
                Section {
                    if model.isLimited {
                        Button("Select More Photos…") { LimitedLibrary.presentPicker() }
                    }
                    Button("Open Photos Access in Settings") {
                        if let url = URL(string: UIApplication.openSettingsURLString) { openURL(url) }
                    }
                } header: {
                    Text("Library Access")
                } footer: {
                    Text(model.isLimited ? "You’ve given access to selected photos only. Search covers just those." : "Search covers your whole library.")
                }
                Section("Privacy") {
                    Label("Search, indexing and text recognition run entirely on this iPhone. No account, no analytics, no network.",
                          systemImage: "lock.shield")
                    Label("Photos that are only in iCloud are never downloaded; their on-device previews are used instead.",
                          systemImage: "icloud.slash")
                }
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } }
            }
            .onAppear { paused = model.progress.state == .paused(.userPaused) }
        }
    }
}

/// `presentLimitedLibraryPicker` needs a UIKit presenter.
enum LimitedLibrary {
    @MainActor static func presentPicker() {
        let scene = UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }.first { $0.activationState == .foregroundActive }
        guard var top = scene?.keyWindow?.rootViewController else { return }
        while let next = top.presentedViewController { top = next }
        PHPhotoLibrary.shared().presentLimitedLibraryPicker(from: top)
    }
}
