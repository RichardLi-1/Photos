import Photos
import SwiftUI

/// First run / no access: explains what the app does with the library before iOS asks.
struct AccessView: View {
    let model: AppModel
    @Environment(\.openURL) private var openURL

    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                Image(systemName: "photo.stack")
                    .font(.system(.largeTitle))
                    .imageScale(.large)
                    .foregroundStyle(.tint)
                    .accessibilityHidden(true)
                Text("Search Your Photos").font(.largeTitle.bold()).multilineTextAlignment(.center)
                Text("Find any photo by describing it — “dog on the beach”, “receipt from March” — or by showing a similar photo. Everything is analysed on this iPhone and never leaves it.")
                    .font(.body).multilineTextAlignment(.center).foregroundStyle(.secondary)
                if model.access == .denied || model.access == .restricted {
                    Text("Photos access is turned off. You can allow all photos, or only the ones you choose, in Settings.")
                        .font(.callout).multilineTextAlignment(.center)
                    Button("Open Settings") {
                        if let url = URL(string: UIApplication.openSettingsURLString) { openURL(url) }
                    }
                    .buttonStyle(.borderedProminent).controlSize(.large)
                } else {
                    Button("Continue") { Task { await model.requestAccess() } }
                        .buttonStyle(.borderedProminent).controlSize(.large)
                    Text("iOS will ask whether to share all photos or only some.").font(.footnote).foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                }
            }
            .padding(24)
            .frame(maxWidth: 520)
            .frame(maxWidth: .infinity)
        }
        .scrollBounceBehavior(.basedOnSize)
    }
}
