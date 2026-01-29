//
//  ContentView.swift
//  Photos
//
//  Created by Richard Li on 2026-01-28.
//

import SwiftUI

struct ContentView: View {
    var body: some View {
        VStack {
            Image(systemName: "globe")
                .imageScale(.large)
                .foregroundStyle(.tint)
            Text("Hello, world!")
            Button {
                print("Heart")

            } label: {
                Label("Like", systemImage: "heart.fill")
            }
        }
        .padding()
    }
    
    func callBackend(){
        print("submitted!")
    }
    
    
    func uploadPhotoEmbeddings(){
}



#Preview {
    ContentView()
}
