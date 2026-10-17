import SwiftUI

/// The app's one "Liquid Glass" material. Every floating surface (search capsule, status pills, match card) uses it, so
/// the look lives in exactly one place.
///
/// Xcode 16 / iOS 18 has no `.glassEffect()`, so this approximates it with a material, a rim highlight and a soft
/// shadow. When the project moves to Xcode 26 / iOS 26, replace the body with
/// `content.glassEffect(.regular.interactive(), in: shape)` (behind `if #available(iOS 26, *)`), and drop the manual
/// highlight/shadow: the system adds refraction, specular rims and adapts to Reduce Transparency itself.
///
/// Accessibility: Reduce Transparency → opaque system background; Increase Contrast → solid, stronger border.
struct GlassBackground<S: InsettableShape>: ViewModifier {
    var shape: S
    var prominent = false  // denser glass for small text (cards), like the web prototype's "regular" variant

    @Environment(\.accessibilityReduceTransparency) private var reduceTransparency
    @Environment(\.colorSchemeContrast) private var contrast
    @Environment(\.colorScheme) private var scheme

    func body(content: Content) -> some View {
        content
            .background {
                if reduceTransparency || contrast == .increased {
                    shape.fill(Color(.secondarySystemBackground))
                } else {
                    shape.fill(prominent ? .regularMaterial : .ultraThinMaterial)
                        .overlay(shape.fill(LinearGradient(colors: [.white.opacity(scheme == .dark ? 0.08 : 0.22), .clear],
                                                           startPoint: .top, endPoint: .center)))
                }
            }
            .overlay {
                if contrast == .increased {
                    shape.strokeBorder(Color.primary.opacity(0.7), lineWidth: 1.5)
                } else {
                    // the rim catching light: bright top-left, faint far side
                    shape.strokeBorder(LinearGradient(colors: [.white.opacity(scheme == .dark ? 0.35 : 0.7), .white.opacity(0.08)],
                                                      startPoint: .topLeading, endPoint: .bottomTrailing), lineWidth: 1)
                }
            }
            .clipShape(shape)
            .shadow(color: .black.opacity(reduceTransparency ? 0.12 : (scheme == .dark ? 0.45 : 0.16)), radius: 18, y: 8)
    }
}

extension View {
    func glassBackground<S: InsettableShape>(in shape: S, prominent: Bool = false) -> some View {
        modifier(GlassBackground(shape: shape, prominent: prominent))
    }
}
