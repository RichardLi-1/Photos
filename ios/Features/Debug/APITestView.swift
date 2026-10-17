//
//  APITestView.swift
//  Photos
//
//  Created by Richard Li on 2026-02-02.
//

// DEBUG-only developer screen for the old localhost backend. It is not reachable from the app's UI and is compiled
// out of Release builds, so the shipping app makes no network calls.
#if DEBUG
import PhotosUI
import SwiftUI
import UIKit

struct APITestView: View {
    @State private var pickerItem: PhotosPickerItem?
    @State private var status: String = "Pick an image"

    var body: some View {
        VStack{
            PhotosPicker(selection: $pickerItem, matching: .images) {
                        Label("Select Image", systemImage: "photo")
                            .padding()
                            .background(Color.blue)
                            .foregroundColor(.white)
                            .cornerRadius(8)
            }
                    
            
            Button(action:{
                print("button tapped bith!")
                callHealthAPI()
            })
            {
                Text("Call Health API")
            }
            Button(action:{print("Call Health API button tapped!")
            }) {
                Text("Call Embeddings API")
            }
            Button(action:{
                print("button tapped!")
                callSearchAPI()
            }) {
                Text("Call Search API")
            }
            Button(action:{print("button tapped!")
            }) {
                Text("Call Ingestion API")
            }
            
            Button("Compute Embedding") {
                Task {
                    do {
                        guard let pickerItem else { return }
                        guard let data = try await pickerItem.loadTransferable(type: Data.self),
                              let uiImage = UIImage(data: data)
                        else { return }

                        guard let cg = PhotoLibrary.cgImage(from: uiImage) else { return }
                        let emb = try await AppServices.shared.encoder().embed(images: [cg])[0]

                        // quick sanity checks
                        let n = sqrt(emb.reduce(0) { $0 + $1*$1 })
                        status = "dim=\(emb.count)  norm=\(String(format: "%.4f", n))  first3=\(emb.prefix(3))"
                        print("embedding dim:", emb.count)
                        print("norm:", n)
                        print("first 10:", Array(emb.prefix(10)))
                    } catch {
                        status = "Error: \(error)"
                        print(error)
                    }
                }
            }
        }
        
    }
    
    
    func callHealthAPI() {
        guard let localUrl = URL(string:"http://127.0.0.1:8000/v1/health")
        else {return}
        
        var request = URLRequest(url: localUrl)
        request.httpMethod = "GET"
        /*request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        let body = [
            "email": "test@test.com",
            "password": "123456"
        ]
         request.httpBody = try? JSONSerialization.data(withJSONObject: body)
         
         */
        
        
        //Create URLRequest Object
        let task = URLSession.shared.dataTask(with: request) { data, response, error in
            guard let data = data
            else { return }
            print(String(data: data, encoding: .utf8)!)
        }
        
        task.resume()
                
    }
    
    func callSearchAPI() {
        guard let localUrl = URL(string:"http://127.0.0.1:8000/v1/search")
        else {return}
        
        var request = URLRequest(url: localUrl)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        let body = [
            "email": "test@test.com",
            "password": "123456"
        ]
         request.httpBody = try? JSONSerialization.data(withJSONObject: body)
         
         
        
        
        //Create URLRequest Object
        let task = URLSession.shared.dataTask(with: request) { data, response, error in
            guard let data = data
            else { return }
            print(String(data: data, encoding: .utf8)!)
        }
        
        task.resume()
                
    }
}


#Preview {
    APITestView()
}
#endif
