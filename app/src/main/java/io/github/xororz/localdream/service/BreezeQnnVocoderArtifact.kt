package io.github.xororz.localdream.service

import android.content.Context
import android.os.Build
import io.github.xororz.localdream.utils.Http
import java.io.File
import java.io.FileOutputStream
import java.security.MessageDigest
import java.util.Locale
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.withContext
import okhttp3.Request
import org.json.JSONObject

object BreezeQnnVocoderArtifact {
    private const val RELEASE_TAG="breeze-qnn-vocoder-sm8850-v2"
    private const val BASE_URL="https://github.com/thaakeno/local-dream/releases/download/"+RELEASE_TAG
    private const val DIR="breeze_qnn_vocoder/v2-sm8850"
    data class Install(val soc:String,val contextFile:File,val lutFile:File)
    sealed class Status {
        object Checking:Status()
        data class Unsupported(val detected:String):Status()
        data class Missing(val soc:String):Status()
        data class Downloading(val soc:String,val received:Long,val total:Long):Status(){
            val progress:Float? get()=if(total>0)(received.toFloat()/total).coerceIn(0f,1f) else null
        }
        data class Ready(val soc:String,val file:File):Status()
        data class Error(val soc:String?,val message:String):Status()
    }
    private val _status=MutableStateFlow<Status>(Status.Checking)
    val status:StateFlow<Status> = _status
    private fun dir(c:Context)=File(c.filesDir,DIR)
    private fun fp():String {
        val p=ArrayList<String>();if(Build.VERSION.SDK_INT>=31)p+=Build.SOC_MODEL.orEmpty()
        p+=Build.HARDWARE.orEmpty();p+=Build.BOARD.orEmpty();p+=Build.DEVICE.orEmpty();p+=Build.PRODUCT.orEmpty()
        return p.joinToString(" ").uppercase(Locale.US)
    }
    fun detectedSoc()=fp().trim().ifBlank{"unknown"}
    fun supportedSoc():String?=if(fp().contains("SM8850"))"SM8850" else null
    fun localInstall(c:Context):Install?{
        val soc=supportedSoc()?:return null;val m=File(dir(c),"installed.json");if(!m.isFile)return null
        return runCatching{
            val j=JSONObject(m.readText());if(j.optInt("version")!=2||j.optString("soc")!=soc)return@runCatching null
            val cf=File(dir(c),j.getString("contextFile")),lf=File(dir(c),j.getString("lutFile"))
            if(!cf.isFile||!lf.isFile||cf.length()!=j.getLong("contextBytes")||lf.length()!=j.getLong("lutBytes"))null else Install(soc,cf,lf)
        }.getOrNull()
    }
    fun localFile(c:Context)=localInstall(c)?.contextFile
    fun refresh(c:Context){val soc=supportedSoc();val i=localInstall(c);_status.value=when{soc==null->Status.Unsupported(detectedSoc());i!=null->Status.Ready(soc,i.contextFile);else->Status.Missing(soc)}}
    private suspend fun get(name:String,bytes:Long,sha:String,target:File,soc:String,base:Long,total:Long){
        val part=File(target.parentFile,target.name+".part");part.delete()
        val req=Request.Builder().url(BASE_URL+"/"+name).get().build()
        Http.client.newCall(req).execute().use{r->
            if(!r.isSuccessful)error("Accelerator download failed (HTTP "+r.code+")")
            val body=r.body?:error("Empty accelerator download");val md=MessageDigest.getInstance("SHA-256");var got=0L
            body.byteStream().use{input->FileOutputStream(part).use{out->val buf=ByteArray(1024*1024);while(true){val n=input.read(buf);if(n<0)break;out.write(buf,0,n);md.update(buf,0,n);got+=n;_status.value=Status.Downloading(soc,base+got,total)}}}
            if(got!=bytes)error("Incomplete accelerator download")
            val actual=md.digest().joinToString(""){"%02x".format(it.toInt() and 0xff)}
            if(actual!=sha.lowercase(Locale.US))error("Checksum mismatch for "+name)
        }
        if(target.exists())target.delete();if(!part.renameTo(target)){part.copyTo(target,true);part.delete()}
    }
    suspend fun download(c:Context)=withContext(Dispatchers.IO){
        val soc=supportedSoc();if(soc==null){_status.value=Status.Unsupported(detectedSoc());return@withContext}
        runCatching{
            val mr=Request.Builder().url(BASE_URL+"/manifest.json").get().build()
            val spec=Http.client.newCall(mr).execute().use{r->if(!r.isSuccessful)error("Accelerator manifest unavailable");JSONObject(r.body?.string()?:error("Empty manifest")).getJSONObject("files").getJSONObject(soc)}
            val cs=spec.getJSONObject("context"),ls=spec.getJSONObject("lut")
            val cn=cs.getString("file"),cb=cs.getLong("bytes"),ch=cs.getString("sha256")
            val ln=ls.getString("file"),lb=ls.getLong("bytes"),lh=ls.getString("sha256");val total=cb+lb
            val d=dir(c);d.mkdirs();val cf=File(d,cn),lf=File(d,ln)
            get(cn,cb,ch,cf,soc,0,total);get(ln,lb,lh,lf,soc,cb,total)
            listOf("LICENSE-Breeze-TTS-2.txt","NOTICE-Breeze-QNN-Vocoder.txt").forEach{name->
                val req=Request.Builder().url(BASE_URL+"/"+name).get().build();Http.client.newCall(req).execute().use{r->if(!r.isSuccessful)error("License download failed");File(d,name).outputStream().use{o->r.body!!.byteStream().use{i->i.copyTo(o)}}}
            }
            File(d,"installed.json").writeText(JSONObject().put("version",2).put("soc",soc).put("contextFile",cn).put("contextBytes",cb).put("contextSha256",ch).put("lutFile",ln).put("lutBytes",lb).put("lutSha256",lh).toString())
            // v1 was a silent-output experiment and is no longer used. Free its
            // ~283 MB automatically only after v2 is fully verified on disk.
            File(c.filesDir,"breeze_qnn_vocoder/v1").deleteRecursively()
            _status.value=Status.Ready(soc,cf)
        }.onFailure{_status.value=Status.Error(soc,it.message?:"Accelerator download failed")}
    }
}
