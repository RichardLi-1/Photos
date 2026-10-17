import com.android.build.api.artifact.SingleArtifact

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

val bundleModels = (findProperty("bundleModels") as String?)?.toBoolean() ?: true

android {
    namespace = "app.photosearch"
    compileSdk = 35
    buildToolsVersion = "35.0.1"

    defaultConfig {
        applicationId = "app.photosearch"
        // 30 (Android 11): MediaStore.createTrashRequest (system-confirmed, recoverable trash) and final scoped storage.
        minSdk = 30
        targetSdk = 35
        versionCode = 1
        versionName = "0.1"
        ndk { abiFilters += listOf("arm64-v8a", "x86_64") } // phones + emulator; drops 32-bit ORT/ML Kit natives
    }

    sourceSets["main"].assets.srcDirs("src/main/assets", *(if (bundleModels) arrayOf("bigassets") else emptyArray()))
    androidResources {
        noCompress += listOf("onnx", "bin") // stored, so the app can mmap them straight out of the APK
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    buildFeatures { compose = true; buildConfig = true }
    testOptions { unitTests.isReturnDefaultValues = true }
    packaging { resources.excludes += "/META-INF/{AL2.0,LGPL2.1}" }
}

dependencies {
    val composeBom = platform("androidx.compose:compose-bom:2024.12.01")
    implementation(composeBom)
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.foundation:foundation")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.material:material-icons-core")
    implementation("androidx.activity:activity-compose:1.9.3")
    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.8.7")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.7")
    implementation("androidx.work:work-runtime-ktx:2.10.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.22.0")
    // On-device OCR. Off by default: ML Kit logs usage metrics to Google (see README). The app has no INTERNET permission.
    implementation("com.google.mlkit:text-recognition:16.0.1")
    debugImplementation("androidx.compose.ui:ui-tooling")

    testImplementation("junit:junit:4.13.2")
    testImplementation("org.json:json:20240303")
}

// Privacy gate: fail the build if any library sneaks a network permission into the merged manifest.
abstract class VerifyNoNetwork : DefaultTask() {
    @get:InputFile abstract val manifest: RegularFileProperty
    @get:OutputFile abstract val stamp: RegularFileProperty

    @TaskAction fun verify() {
        val text = manifest.get().asFile.readText()
        val banned = listOf("android.permission.INTERNET", "android.permission.ACCESS_NETWORK_STATE", "android.permission.ACCESS_WIFI_STATE")
        val found = banned.filter { Regex("""uses-permission[^>]*"$it"""").containsMatchIn(text) }
        check(found.isEmpty()) { "Merged manifest requests network permissions: $found" }
        stamp.get().asFile.writeText("ok")
    }
}

androidComponents {
    onVariants { variant ->
        val task = tasks.register<VerifyNoNetwork>("verifyNoNetwork${variant.name.replaceFirstChar { it.uppercase() }}") {
            manifest.set(variant.artifacts.get(SingleArtifact.MERGED_MANIFEST))
            stamp.set(layout.buildDirectory.file("verifyNoNetwork/${variant.name}.txt"))
        }
        val assemble = "assemble${variant.name.replaceFirstChar { it.uppercase() }}"
        tasks.matching { it.name == assemble }.configureEach { dependsOn(task) }
    }
}
