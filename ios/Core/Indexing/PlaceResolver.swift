import CoreLocation

/// Reverse-geocodes photo locations into place names ("Paris, Île-de-France, France").
///
/// PRIVACY: CLGeocoder sends coordinates to Apple's servers. This is the only networked feature in the app, it is OFF
/// by default and only runs after the user turns on Settings → "Look Up Place Names". Results are cached on a ~1 km
/// grid so each area is looked up once, and requests are throttled well under Apple's rate limits.
actor PlaceResolver {
    static let settingKey = "placeLookupEnabled"
    private let geocoder = CLGeocoder()
    private var cache: [String: String?] = [:]

    static var isEnabled: Bool { UserDefaults.standard.bool(forKey: settingKey) }

    /// Throws when offline / rate limited (the caller retries later); `nil` means "no named place here".
    func place(lat: Double, lon: Double) async throws -> String? {
        let key = String(format: "%.2f,%.2f", lat, lon)
        if let hit = cache[key] { return hit }
        try await Task.sleep(for: .seconds(1.5))  // throttle
        let marks: [CLPlacemark]
        do {
            marks = try await geocoder.reverseGeocodeLocation(CLLocation(latitude: lat, longitude: lon))
        } catch CLError.geocodeFoundNoResult {
            marks = []
        }
        let name = marks.first.map { m in
            [m.locality, m.administrativeArea, m.country].compactMap { $0 }.reduce(into: [String]()) { if !$0.contains($1) { $0.append($1) } }
                .joined(separator: ", ")
        }.flatMap { $0.isEmpty ? nil : $0 }
        cache[key] = name
        return name
    }
}
