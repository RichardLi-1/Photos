//
//  PhotosApp.swift
//  Photos
//
//  Created by Richard Li on 2026-01-28.
//

import SwiftUI

@main
struct PhotosApp: App {
    init() {
        BackgroundIndexing.register()  // must happen before launch finishes
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
