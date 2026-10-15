//
//  ContentView.swift
//  Photos
//
//  Created by Richard Li on 2026-01-28.
//

import SwiftUI

/// Root: full-bleed photo grid, floating glass search capsule at the bottom, settings in the toolbar.
struct ContentView: View {
    @State private var model = AppModel()
    @State private var path: [String] = []
    @State private var showSettings = false
    @Environment(\.scenePhase) private var scenePhase
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        NavigationStack(path: $path) {
            Group {
                if model.hasAccess {
                    library
                } else {
                    AccessView(model: model)
                }
            }
            .navigationTitle(title)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button { showSettings = true } label: { Label("Settings", systemImage: "gearshape") }
                }
                if model.isLimited {
                    ToolbarItem(placement: .topBarLeading) {
                        Button { LimitedLibrary.presentPicker() } label: { Label("Select More Photos", systemImage: "plus.circle") }
                    }
                }
            }
            .navigationDestination(for: String.self) { PhotoDetailView(model: model, localID: $0) }
        }
        .sheet(isPresented: $showSettings) { SettingsView(model: model) }
        .sensoryFeedback(.error, trigger: model.searchFailures)
        .task {
            await model.start()
            #if DEBUG
            await SelfTest.runIfRequested()
            #endif
        }
        .onChange(of: scenePhase) { _, phase in
            if phase == .background { BackgroundIndexing.schedule() }
            if phase == .active { Task { await model.start() } }
        }
    }

    private var library: some View {
        PhotoGrid(model: model) { path.append($0) }
            .overlay {
                if model.items.count == 0 && model.query != nil && !model.isSearching {
                    ContentUnavailableView.search(text: model.searchText)
                }
            }
            .safeAreaInset(edge: .bottom, spacing: 0) {
                VStack(spacing: 10) {
                    StatusPills(model: model) { showSettings = true }
                    SearchCapsule(model: model)
                        .frame(maxWidth: 720)
                }
                .padding(.horizontal, 16)
                .padding(.bottom, 8)
                .animation(reduceMotion ? nil : .smooth, value: model.status)
            }
    }

    private var title: LocalizedStringKey {
        switch model.query {
        case nil: "Library"
        case .similar: "Similar Photos"
        default: "Results"
        }
    }
}

#Preview {
    ContentView()
}
