package io.github.xororz.localdream.data

/**
 * Natural everyday read-aloud passages for recording a voice reference.
 * All text stays on the device. These are sample transcripts, not model voices.
 * Users can edit the exact transcript to match what they actually recorded.
 */
data class BreezeVoiceScript(val code: String, val language: String, val text: String)

object BreezeVoiceScripts {
    val options: List<BreezeVoiceScript> = listOf(
        BreezeVoiceScript("en", "English",
            "I went out for a walk this morning before the streets got busy. The air was a little cool, but the sun was already coming through the trees. On the way home, I stopped for coffee and ran into a neighbor I hadn't seen in a while. We talked for a few minutes, then I headed back."),
        BreezeVoiceScript("de", "Deutsch",
            "Heute Morgen war ich noch kurz draußen, bevor es auf den Straßen voll wurde. Es war etwas kühl, aber die Sonne kam schon zwischen den Häusern durch. Auf dem Rückweg habe ich beim Bäcker angehalten und zufällig einen alten Bekannten getroffen. Wir haben uns kurz unterhalten, danach bin ich nach Hause gegangen."),
        BreezeVoiceScript("tr", "Türkçe",
            "Bu sabah sokaklar kalabalıklaşmadan biraz yürüyüşe çıktım. Hava serindi ama güneş yavaş yavaş kendini göstermeye başlamıştı. Eve dönerken fırına uğradım ve uzun zamandır görmediğim bir tanıdığımla karşılaştım. Birkaç dakika sohbet ettik, sonra ekmeğimi alıp eve gittim."),
        BreezeVoiceScript("fr", "Français",
            "Ce matin, je suis sorti marcher un peu avant que les rues deviennent trop animées. Il faisait encore frais, mais le soleil commençait à éclairer les façades. En rentrant, je me suis arrêté à la boulangerie et j'ai croisé un voisin que je n'avais pas vu depuis longtemps. On a discuté quelques minutes."),
        BreezeVoiceScript("es", "Español",
            "Esta mañana salí a dar una vuelta antes de que hubiera mucha gente en la calle. Hacía un poco de fresco, pero ya empezaba a salir el sol. De camino a casa pasé por la panadería y me encontré con un vecino al que no veía desde hacía tiempo. Charlamos unos minutos y seguí mi camino."),
        BreezeVoiceScript("it", "Italiano",
            "Stamattina sono uscito a fare due passi prima che le strade diventassero affollate. Faceva ancora un po' fresco, ma il sole cominciava a scaldare l'aria. Tornando a casa mi sono fermato al panificio e ho incontrato un vecchio conoscente. Abbiamo parlato qualche minuto, poi sono ripartito."),
        BreezeVoiceScript("ja", "日本語",
            "今朝は道が混む前に、少し散歩に出かけました。まだ空気はひんやりしていましたが、木の間から朝日が差し込んでいて、気持ちのいい時間でした。帰り道に近所のパン屋に寄ったら、久しぶりに知り合いに会いました。少し立ち話をしてから、家に帰りました。"),
        BreezeVoiceScript("zh", "中文",
            "今天早上，我趁街上还不太热闹的时候出去走了一会儿。天气有点凉，不过太阳已经慢慢升起来了。回家的路上，我顺便去买了点早餐，还碰到了一位很久没见的邻居。我们聊了几分钟，然后就各自回家了。"),
        BreezeVoiceScript("ko", "한국어",
            "오늘 아침에는 거리가 붐비기 전에 잠깐 산책을 나갔어요. 공기는 조금 쌀쌀했지만 나무 사이로 햇빛이 들어와서 기분이 좋았어요. 집에 돌아오는 길에 빵집에 들렀다가 오랜만에 아는 이웃을 만났어요. 잠깐 이야기를 나눈 뒤에 집으로 돌아왔습니다.")
    )
    fun find(code: String): BreezeVoiceScript =
        options.firstOrNull { it.code == code } ?: options.first()
}
