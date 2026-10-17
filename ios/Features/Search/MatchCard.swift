import SwiftUI

/// Small card beside a tapped photo: what it has in common with the current search (web/src/components/MatchCard.tsx).
struct MatchCard: View {
    let model: AppModel
    let localID: String
    var onOpen: (String) -> Void

    @State private var explanation: Explanation?
    @State private var loaded = false
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            if let e = explanation {
                Text(title(e))
                    .font(.footnote.weight(.semibold))
                    .foregroundStyle(.secondary)
                    .accessibilityAddTraits(.isHeader)
                if e.shared.isEmpty {
                    Text("Nothing specific — mostly the overall look.").font(.subheadline).foregroundStyle(.secondary)
                } else {
                    FlowLayout(spacing: 6) {
                        ForEach(e.shared, id: \.self) { Chip(text: $0) }
                    }
                    .accessibilityElement(children: .combine)
                }
                if !e.textHits.isEmpty {
                    Label("“\(e.textHits.joined(separator: ", "))” is written in the photo", systemImage: "text.viewfinder")
                        .font(.subheadline)
                }
                if !e.placeHits.isEmpty {
                    Label("“\(e.placeHits.joined(separator: ", "))” is in the place name", systemImage: "mappin.and.ellipse")
                        .font(.subheadline)
                }
                if let line = detailLine(e) {
                    Text(line).font(.caption).foregroundStyle(.secondary)
                }
            } else if loaded {
                Text("This photo isn’t indexed yet.").font(.subheadline).foregroundStyle(.secondary)
            } else {
                ProgressView("Looking…").font(.subheadline)
            }
            ViewThatFits(in: .horizontal) {
                HStack(spacing: 12) { buttons }
                VStack(alignment: .leading, spacing: 4) { buttons }
            }
        }
        .padding(16)
        .frame(minWidth: 240, idealWidth: 280, maxWidth: 340, alignment: .leading)
        .task(id: localID) {
            explanation = await model.explain(localID)
            loaded = true
        }
    }

    @ViewBuilder private var buttons: some View {
        Button {
            model.moreLikeThis(localID)
        } label: {
            Label("More Like This", systemImage: "square.grid.2x2")
                .frame(minHeight: 44)
        }
        .disabled(explanation == nil)
        Button {
            dismiss()
            onOpen(localID)
        } label: {
            Label("Open", systemImage: "arrow.up.left.and.arrow.down.right")
                .frame(minHeight: 44)
        }
    }

    private func title(_ e: Explanation) -> LocalizedStringResource {
        if e.overall { return "Overall look" }
        switch (e.mode, model.query) {
        case (.text, .text(let q)?): return "Matches “\(q)”"
        case (.similar, _): return "In common with that photo"
        case (.photo, _): return "In common with your photo"
        default: return "In this photo"
        }
    }

    private func detailLine(_ e: Explanation) -> String? {
        let parts = [e.place, e.date?.formatted(date: .abbreviated, time: .omitted)].compactMap { $0 }
        return parts.isEmpty ? nil : parts.joined(separator: " · ")
    }
}

struct Chip: View {
    let text: String
    var body: some View {
        Text(text)
            .font(.subheadline)
            .padding(.horizontal, 10)
            .padding(.vertical, 4)
            .background(Capsule().fill(Color(.tertiarySystemFill)))
    }
}

/// Wraps children onto as many lines as needed (chips grow with Dynamic Type).
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
