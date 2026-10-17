import SwiftUI

/// The inspector beside a selected photo: a large preview and why it matched (web/src/components/MatchCard.tsx,
/// ios/Features/Search/MatchCard.swift), plus More Like This and Same Place.
struct MatchCard: View {
    let model: AppModel
    let hit: SearchHit
    @State private var explanation: Explanation?
    @State private var loaded = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                Thumbnail(localID: hit.localID, maxSide: 1400, contentMode: .fit)
                    .frame(maxWidth: .infinity, minHeight: 160)
                    .clipShape(RoundedRectangle(cornerRadius: 10))
                    .onTapGesture(count: 2) { model.open(hit) }

                VStack(alignment: .leading, spacing: 10) {
                    if let e = explanation {
                        Text(title(e)).font(.footnote.weight(.semibold)).foregroundStyle(.secondary)
                            .accessibilityAddTraits(.isHeader)
                        if e.shared.isEmpty {
                            Text("Nothing specific — mostly the overall look.").foregroundStyle(.secondary)
                        } else {
                            FlowLayout(spacing: 6) { ForEach(e.shared, id: \.self) { Chip(text: $0) } }
                                .accessibilityElement(children: .combine)
                        }
                        if !e.textHits.isEmpty {
                            Label("“\(e.textHits.joined(separator: ", "))” is written in the photo", systemImage: "text.viewfinder")
                        }
                        if !e.placeHits.isEmpty {
                            Label("“\(e.placeHits.joined(separator: ", "))” is in the place name", systemImage: "mappin.and.ellipse")
                        }
                        if let line = detailLine(e) { Text(line).font(.caption).foregroundStyle(.secondary) }
                    } else if loaded {
                        Text("This photo isn’t indexed yet.").foregroundStyle(.secondary)
                    } else {
                        ProgressView().controlSize(.small)
                    }
                    if let score = hit.score {
                        Text("Similarity \(score, format: .number.precision(.fractionLength(3)))")
                            .font(.caption.monospacedDigit()).foregroundStyle(.tertiary)
                    }
                }
                .padding(14)
                .glassBackground(in: RoundedRectangle(cornerRadius: 14), prominent: true)

                VStack(alignment: .leading, spacing: 8) {
                    Button { model.moreLikeThis(hit.id) } label: { Label("More Like This", systemImage: "square.grid.2x2") }
                    if model.placeAvailable {
                        Button { model.samePlace(hit.id) } label: { Label("Same Place", systemImage: "mappin.and.ellipse") }
                            .help("Other photos of this room or place, using a small model we trained on hotel photos")
                    }
                    HStack {
                        Button { model.open(hit) } label: { Label("Open", systemImage: "arrow.up.forward.app") }
                        if PhotoRef(localID: hit.localID)?.fileURL != nil {
                            Button { model.revealInFinder(hit) } label: { Label("Show in Finder", systemImage: "folder") }
                        }
                    }
                }
                .buttonStyle(.bordered)
                .controlSize(.large)
            }
            .padding(14)
        }
        .task(id: hit.id) {
            explanation = nil
            loaded = false
            explanation = await model.explain(hit.id)
            loaded = true
        }
    }

    private func title(_ e: Explanation) -> String {
        if e.overall { return String(localized: "Overall look") }
        switch (e.mode, model.mode) {
        case (.text, .text(let q)): return String(localized: "Matches “\(q)”")
        case (.similar, _): return String(localized: "In common with that photo")
        case (.photo, _): return String(localized: "In common with your photo")
        default: return String(localized: "In this photo")
        }
    }

    private func detailLine(_ e: Explanation) -> String? {
        let parts = [PhotoRef(localID: hit.localID)?.displayName, e.place, e.date?.formatted(date: .abbreviated, time: .shortened)].compactMap { $0 }
        return parts.isEmpty ? nil : parts.joined(separator: " · ")
    }
}

struct Chip: View {
    let text: String
    var body: some View {
        Text(text)
            .padding(.horizontal, 10).padding(.vertical, 4)
            .background(Capsule().fill(.quaternary))
    }
}

/// Wraps children onto as many lines as needed (same layout as the iOS app's chips).
struct FlowLayout: Layout {
    var spacing: CGFloat = 8

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let rows = arrange(proposal.width ?? .infinity, subviews)
        return CGSize(width: rows.map(\.width).max() ?? 0, height: rows.last.map { $0.y + $0.height } ?? 0)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        for row in arrange(bounds.width, subviews) {
            var x = bounds.minX
            for i in row.items {
                let size = subviews[i].sizeThatFits(.init(width: bounds.width, height: nil))
                subviews[i].place(at: CGPoint(x: x, y: bounds.minY + row.y), proposal: .init(size))
                x += size.width + spacing
            }
        }
    }

    private struct Row { var items: [Int] = []; var width: CGFloat = 0; var height: CGFloat = 0; var y: CGFloat = 0 }

    private func arrange(_ maxWidth: CGFloat, _ subviews: Subviews) -> [Row] {
        var rows: [Row] = [Row()]
        for i in subviews.indices {
            let size = subviews[i].sizeThatFits(.init(width: maxWidth, height: nil))
            if !rows[rows.count - 1].items.isEmpty && rows[rows.count - 1].width + spacing + size.width > maxWidth {
                let last = rows[rows.count - 1]
                rows.append(Row(y: last.y + last.height + spacing))
            }
            var row = rows[rows.count - 1]
            row.width += (row.items.isEmpty ? 0 : spacing) + min(size.width, maxWidth)
            row.height = max(row.height, size.height)
            row.items.append(i)
            rows[rows.count - 1] = row
        }
        return rows.filter { !$0.items.isEmpty }
    }
}

/// The app's one glass material (macOS counterpart of ios/Features/Shared/GlassBackground.swift). Xcode 16 has no
/// `.glassEffect()`; with Xcode 26 swap the body for `content.glassEffect(.regular, in: shape)` behind #available.
struct GlassBackground<S: InsettableShape>: ViewModifier {
    var shape: S
    var prominent = false
    @Environment(\.accessibilityReduceTransparency) private var reduceTransparency
    @Environment(\.colorSchemeContrast) private var contrast
    @Environment(\.colorScheme) private var scheme

    func body(content: Content) -> some View {
        content
            .background {
                if reduceTransparency || contrast == .increased {
                    shape.fill(Color(nsColor: .windowBackgroundColor))
                } else {
                    shape.fill(prominent ? .regularMaterial : .thinMaterial)
                }
            }
            .overlay {
                shape.strokeBorder(contrast == .increased ? AnyShapeStyle(Color.primary.opacity(0.7))
                                   : AnyShapeStyle(LinearGradient(colors: [.white.opacity(scheme == .dark ? 0.25 : 0.6), .white.opacity(0.05)],
                                                                  startPoint: .topLeading, endPoint: .bottomTrailing)),
                                   lineWidth: contrast == .increased ? 1.5 : 1)
            }
            .clipShape(shape)
    }
}

extension View {
    func glassBackground<S: InsettableShape>(in shape: S, prominent: Bool = false) -> some View {
        modifier(GlassBackground(shape: shape, prominent: prominent))
    }
}
