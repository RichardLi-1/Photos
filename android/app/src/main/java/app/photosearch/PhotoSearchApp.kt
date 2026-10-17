package app.photosearch

import android.app.Application

class PhotoSearchApp : Application() {
    override fun onCreate() {
        super.onCreate()
        AppGraph.get(this)
    }
}
