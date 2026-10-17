import Photos
import SwiftUI

/// A single photo, pushed onto the navigation stack.
struct PhotoDetailView: View {
    let model: AppModel
    let localID: String
    @Environment(\.dismiss) private var dismiss

    private var asset: PHAsset? { PhotoLibrary.shared.assets([localID]).first }

    var body: some View {
        Group {
            if let asset {
                Thumbnail(asset: asset, contentMode: .fit, targetSide: 1600)
                    .accessibilityElement()
                    .accessibilityLabel(PhotoCell.label(for: asset))
                    .accessibilityAddTraits(.isImage)
            } else {
                ContentUnavailableView("Photo Unavailable", systemImage: "photo")
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(.systemBackground))
        .navigationTitle(asset?.creationDate?.formatted(date: .abbreviated, time: .omitted) ?? "")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .bottomBar) {
                Button {
                    model.moreLikeThis(localID)
                    dismiss()
                } label: {
                    Label("More Like This", systemImage: "square.grid.2x2")
                }
            }
        }
    }
}
