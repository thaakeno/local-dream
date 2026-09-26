package io.github.xororz.localdream.data.db

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import kotlinx.coroutines.flow.Flow

@Dao
interface HistoryCollectionDao {
    @Query(
        """
        SELECT c.id, c.name, c.createdAt, COUNT(i.historyId) AS itemCount
        FROM history_collections c
        LEFT JOIN history_collection_items i ON i.collectionId = c.id
        GROUP BY c.id, c.name, c.createdAt
        ORDER BY c.createdAt DESC, c.id DESC
        """,
    )
    fun observeCollections(): Flow<List<HistoryCollectionSummary>>

    @Insert(onConflict = OnConflictStrategy.ABORT)
    suspend fun insertCollection(collection: HistoryCollectionEntity): Long

    @Query("UPDATE history_collections SET name = :name WHERE id = :id")
    suspend fun renameCollection(id: Long, name: String): Int

    @Query("DELETE FROM history_collections WHERE id = :id")
    suspend fun deleteCollection(id: Long): Int

    @Insert(onConflict = OnConflictStrategy.IGNORE)
    suspend fun addItems(items: List<HistoryCollectionItemEntity>)

    @Query(
        "DELETE FROM history_collection_items WHERE collectionId = :collectionId AND historyId IN (:historyIds)",
    )
    suspend fun removeItems(collectionId: Long, historyIds: List<Long>)

    @Query(
        "SELECT collectionId FROM history_collection_items WHERE historyId = :historyId ORDER BY collectionId",
    )
    fun observeCollectionIdsForHistory(historyId: Long): Flow<List<Long>>
}
