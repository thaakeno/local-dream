package io.github.xororz.localdream.data.db

import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.Index
import androidx.room.PrimaryKey

@Entity(
    tableName = "history_collections",
    indices = [Index(value = ["name"], unique = true)],
)
data class HistoryCollectionEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val name: String,
    val createdAt: Long,
)

@Entity(
    tableName = "history_collection_items",
    primaryKeys = ["collectionId", "historyId"],
    foreignKeys = [
        ForeignKey(
            entity = HistoryCollectionEntity::class,
            parentColumns = ["id"],
            childColumns = ["collectionId"],
            onDelete = ForeignKey.CASCADE,
        ),
        ForeignKey(
            entity = HistoryEntity::class,
            parentColumns = ["id"],
            childColumns = ["historyId"],
            onDelete = ForeignKey.CASCADE,
        ),
    ],
    indices = [
        Index(value = ["collectionId"]),
        Index(value = ["historyId"]),
    ],
)
data class HistoryCollectionItemEntity(
    val collectionId: Long,
    val historyId: Long,
    val addedAt: Long,
)

data class HistoryCollectionSummary(
    val id: Long,
    val name: String,
    val createdAt: Long,
    val itemCount: Int,
)
