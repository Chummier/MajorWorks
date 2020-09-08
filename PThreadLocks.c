#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

pthread_cond_t rwlock;

/* An implementation of read/write mutex locks in PThreads
The locks stop parallel programs with multiple threads from having
two threads try and read or write to the same piece of data
at the same time
*/

typedef struct newLock{
    int readers;
    int isWriting;
    int pendingWriters;

    pthread_mutex_t lock;
    pthread_cond_t readers_proceed;
    pthread_cond_t writer_proceed;
}mylock;

mylock manualLock;

static void compResults(char *string, int rc) {
  if (rc) {
    printf("Error on : %s, rc=%d",
           string, rc);
    exit(EXIT_FAILURE);
  }
  return;
}

int myrwlock_rdlock(mylock s){
    int res = 0;
    s.readers++; // Add a new reader
        res = pthread_mutex_lock(&s.lock); // Only let one reader check
        while (s.isWriting){ // While something's writing, wait
            res = pthread_cond_wait(&s.writer_proceed, &s.lock);
        }
        s.isWriting = 0; // When it's done, nothing's writing
        res = pthread_cond_broadcast(&s.readers_proceed); // So readers can go
        res = pthread_mutex_unlock(&s.lock); // Let other readers check
    s.readers--;
    return res;
}

int myrwlock_wrlock(mylock s){
    int res = 0;
    s.pendingWriters++; // This writer is now pending
        res = pthread_mutex_lock(&s.lock); // Only let one pending writer check
        while(s.isWriting){ // If something's writing already, wait
            res = pthread_cond_wait(&s.readers_proceed, &s.lock);
        }
        res = pthread_cond_signal(&s.writer_proceed); // Else, let the writer go
        res = pthread_cond_broadcast(&s.readers_proceed); // Tell readers to stop
        res = pthread_mutex_unlock(&s.lock); // Let other writers check now
    s.pendingWriters--; // No longer pending; it's writing now
    s.isWriting = 1; 
    return res;
}

int myrwlock_unlock(mylock s){

    int res = 0;

    res = pthread_cond_broadcast(&s.readers_proceed);
    res = pthread_cond_broadcast(&s.writer_proceed);
    return res;
}

int myrwlock_init(mylock s){
    s.readers = 0;
    s.isWriting = 0;
    s.pendingWriters = 0;

    int res = 0;

    res = pthread_mutex_init(&s.lock, NULL);
    res = pthread_cond_init(&s.readers_proceed, NULL);
    res = pthread_cond_init(&s.writer_proceed, NULL);
    return res;
}

int myrwlock_destroy(mylock s){

    int res = 0;

    res = pthread_cond_destroy(&s.readers_proceed);
    res = pthread_cond_destroy(&s.writer_proceed);
    return res;
}

void *rdlockThread(void *arg)
{
  int rc;

  printf("Entered thread, getting read lock\n");
  //rc = pthread_rwlock_rdlock(&rwlock);
  rc = myrwlock_rdlock(manualLock);
  compResults("pthread_rwlock_rdlock()\n", rc);
  printf("got the rwlock read lock\n");

  sleep(5);

  printf("unlock the read lock\n");
  //rc = pthread_rwlock_unlock(&rwlock);
  rc = myrwlock_unlock(manualLock);
  compResults("pthread_rwlock_unlock()\n", rc);
  printf("Secondary thread unlocked\n");
  return NULL;
}

void *wrlockThread(void *arg)
{
  int rc;

  printf("Entered thread, getting write lock\n");
  //rc = pthread_rwlock_wrlock(&rwlock);
  rc = myrwlock_wrlock(manualLock);
  compResults("pthread_rwlock_wrlock()\n", rc);

  printf("Got the rwlock write lock, now unlock\n");
  //rc = pthread_rwlock_unlock(&rwlock);
  rc = myrwlock_unlock(manualLock);
  compResults("pthread_rwlock_unlock()\n", rc);
  printf("Secondary thread unlocked\n");
  return NULL;
}



int main(int argc, char **argv)
{
  int                   rc=0;
  pthread_t             thread, thread1, thread2, thread3;

  printf("Enter test case - %s\n", argv[0]);

  printf("Main, initialize the read write lock\n");
  //rc = pthread_rwlock_init(&rwlock, NULL);
  rc = myrwlock_init(manualLock);
  compResults("pthread_rwlock_init()\n", rc);

  printf("Main, grab a read lock\n");
  //rc = pthread_rwlock_rdlock(&rwlock);
  rc = myrwlock_rdlock(manualLock);
  compResults("pthread_rwlock_rdlock()\n",rc);

  printf("Main, grab the same read lock again\n");
  //rc = pthread_rwlock_rdlock(&rwlock);
  rc = myrwlock_rdlock(manualLock);
  compResults("pthread_rwlock_rdlock() second\n", rc);

  printf("Main, grab a write lock\n");
  //rc = pthread_rwlock_rdlock(&rwlock);
  rc = myrwlock_wrlock(manualLock);
  compResults("pthread_rwlock_rdlock() second\n", rc);

  printf("Main, grab the same read lock again again\n");
  //rc = pthread_rwlock_rdlock(&rwlock);
  rc = myrwlock_rdlock(manualLock);
  compResults("pthread_rwlock_rdlock() second\n", rc);

  printf("Main, create the read lock thread\n");
  rc = pthread_create(&thread, NULL, rdlockThread, NULL);
  compResults("pthread_create\n", rc);

  printf("Main - unlock the first read lock\n");
  //rc = pthread_rwlock_unlock(&rwlock);
  rc = myrwlock_unlock(manualLock);
  compResults("pthread_rwlock_unlock()\n", rc);

  printf("Main, create the write lock thread\n");
  rc = pthread_create(&thread1, NULL, wrlockThread, NULL);
  compResults("pthread_create\n", rc);

  printf("Main, create the read lock thread 2\n");
  rc = pthread_create(&thread2, NULL, rdlockThread, NULL);
  compResults("pthread_create\n", rc);

  printf("Main, create the read lock thread 3\n");
  rc = pthread_create(&thread3, NULL, rdlockThread, NULL);
  compResults("pthread_create\n", rc);

  sleep(5);
  printf("Main - unlock the second read lock\n");
  //rc = pthread_rwlock_unlock(&rwlock);
  rc = myrwlock_unlock(manualLock);
  compResults("pthread_rwlock_unlock()\n", rc);

  sleep(5);
  printf("Main - unlock the write lock \n");
  //rc = pthread_rwlock_unlock(&rwlock);
  rc = myrwlock_unlock(manualLock);
  compResults("pthread_rwlock_unlock()\n", rc);

  sleep(5);
  printf("Main - unlock the third read lock \n");
  //rc = pthread_rwlock_unlock(&rwlock);
  rc = myrwlock_unlock(manualLock);
  compResults("pthread_rwlock_unlock()\n", rc);

  printf("Main, wait for the threads\n");
  rc = pthread_join(thread, NULL);
  compResults("pthread_join\n", rc);

  rc = pthread_join(thread1, NULL);
  compResults("pthread_join\n", rc);

  //rc = pthread_rwlock_destroy(&rwlock);
  rc = myrwlock_destroy(manualLock);
  compResults("pthread_rwlock_destroy()\n", rc);

  printf("Main completed\n");
  return 0;
}
