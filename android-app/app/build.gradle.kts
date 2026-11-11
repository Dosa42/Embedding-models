plugins {
    id("com.android.application")
}

android {
    namespace = "com.ayrvokan.embeddingmodels"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.ayrvokan.embeddingmodels"
        minSdk = 26
        targetSdk = 36
        versionCode = 1
        versionName = "0.1.0"
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

dependencies {
    implementation(fileTree(mapOf("dir" to "libs", "include" to listOf("*.jar", "*.aar"))))
}
