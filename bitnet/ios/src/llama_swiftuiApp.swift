import Darwin
import Foundation
import SwiftUI
import UIKit

@main
struct llama_swiftuiApp: App {
    var body: some Scene {
        WindowGroup {
            if ProcessInfo.processInfo.environment["BITNET_CANDIDATE_CONFIG"] != nil {
                Text("Running BitNet physical-device experiment")
                    .task {
                        UIApplication.shared.isIdleTimerDisabled = true
                        UIDevice.current.isBatteryMonitoringEnabled = true
                        await Task.detached(priority: .userInitiated) {
                            // Keep model/Metal initialization outside scene creation.
                            let code = bitnet_candidate_run()
                            fflush(stdout)
                            fflush(stderr)
                            exit(code)
                        }.value
                    }
            } else {
                ContentView()
            }
        }
    }
}
