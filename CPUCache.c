#include "tips.h"
#include <stdio.h>

/*
I've deleted the parts of this code that I didn't write
This program simulates a CPU cache for any given
MIPS program input
*/

/*

  @param addr 32-bit byte address
  @param data a pointer to a SINGLE word (32-bits of data)
  @param we   if we == READ, then data used to return
              information back to CPU

              if we == WRITE, then data used to
              update Cache/DRAM
*/
void accessMemory(address addr, word* data, WriteEnable we)
{
  if(assoc == 0) {
    accessDRAM(addr, (byte*)data, WORD_SIZE, we);
    return;
  }

  // https://stackoverflow.com/questions/3064926/how-to-write-log-base2-in-c-c

  unsigned int temp = 0;

  unsigned int offsetBits = block_size;
  while (offsetBits >>= 1){ ++temp; }
  offsetBits = temp;

  temp = 0;

  unsigned int indexBits = set_count;
  while (indexBits >>= 1){ ++temp; }
  indexBits = temp;

  unsigned int num = 1;
  for (int i = 0; i < offsetBits; i++){ num *= 2; }
  unsigned int offset = addr&(num - 1);

  num = 1;
  for (int i = 0; i < indexBits; i++){ num *= 2; }
  unsigned int index  = (addr >> offsetBits)&(num - 1);

  unsigned int tag = addr >> (offsetBits+indexBits);

  int foundTag = -1;
  int lruBlock = 0;
  for (int i = 0; i < assoc; i++){
    if (cache[index].block[i].tag == tag){
      foundTag = i;
    }
  }

  // CHECKLIST
  // handles associativity CHECK
  // handles block sizes CHECK
  // handles indices CHECK
  // handles lw CHECK
  // handles sw CHECK
  // handles write through CHECK
  // handles write back CHECK
  // handles forward jump statements CHECK
  // handles lru I GUESS
  // handles random CHECK

  // doesn't exactly work when: associativity is 5, one set, block size is 32 and the program is doing reading
  // and writing to the cache. At the end, some of the register values are slightly off.
  // write through has problems with associativity of 5, one, and block size = 32 too with lw and sw programs

  // If it's gonna be a miss, see if there are open blocks in the set. Defaults to 0 if it finds nothing

  // Lru system goes top to bottom in a set. In empty cache it starts at block[0], then next would be block[1]
  // then technically block[0] is now the lru so it replaces that, then block[1], and so on
  if (foundTag == -1){
    if (policy == RANDOM){
      lruBlock = randomint(assoc);
    } else {
      for (int i = 0; i < assoc; i++){
        if (cache[index].block[i].lru.value != 1){
          lruBlock = i;
          break;
        }
      }
    }
  } else {
    lruBlock = foundTag;
  }


  if (we == READ){
    // Cache read miss OKAY
    if (foundTag == -1){

      highlight_block(index, lruBlock);
      highlight_offset(index, lruBlock, offset, MISS);

      // If block is dirty, write its data back to DRAM first
      if (cache[index].block[lruBlock].dirty == DIRTY){
        unsigned int oldAddr = cache[index].block[lruBlock].tag << indexBits;
        oldAddr = oldAddr | index;
        oldAddr = oldAddr << (offsetBits);

        switch(block_size){
          case 4: accessDRAM(oldAddr, (byte*)cache[index].block[lruBlock].data, WORD_SIZE, WRITE);
                break;
          case 8: accessDRAM(oldAddr, (byte*)cache[index].block[lruBlock].data, DOUBLEWORD_SIZE, WRITE);
                break;
          case 16:accessDRAM(oldAddr, (byte*)cache[index].block[lruBlock].data, QUADWORD_SIZE, WRITE);
                break;
          case 32:accessDRAM(oldAddr, (byte*)cache[index].block[lruBlock].data, OCTWORD_SIZE, WRITE);
        }
      }
      
      // Get data from DRAM OKAY
      switch(block_size){
        case 4: accessDRAM(addr-offset, (byte*)data, WORD_SIZE, READ);
                break;
        case 8: accessDRAM(addr-offset, (byte*)data, DOUBLEWORD_SIZE, READ);
                break;
        case 16:accessDRAM(addr-offset, (byte*)data, QUADWORD_SIZE, READ);
                break;
        case 32:accessDRAM(addr-offset, (byte*)data, OCTWORD_SIZE, READ);
      }

      // Copy data to cache block, update tag and flag bits OKAY
      memcpy(cache[index].block[lruBlock].data, data, block_size);
      cache[index].block[lruBlock].tag = tag;
      cache[index].block[lruBlock].valid = VALID;
      cache[index].block[lruBlock].dirty = VIRGIN;

      for (int i = 0; i < assoc; i++){
        if (cache[index].block[i].lru.value == 1){
          cache[index].block[i].lru.value = 0;
        }
      }
      cache[index].block[lruBlock].lru.value = 1;

      memcpy(data, cache[index].block[lruBlock].data+offset, 4);
    // Cache read hit OKAY
    } else {
      highlight_offset(index, lruBlock, offset, HIT);
      memcpy(data, cache[index].block[lruBlock].data+offset, 4);
    }
  } else {
    // Cache write miss OKAY
    if (foundTag == -1){
      
      highlight_offset(index, lruBlock, offset, MISS);
      // Write through miss OKAY
      if (memory_sync_policy == WRITE_THROUGH){
        switch(block_size){
        case 4: accessDRAM(addr, (byte*)data, WORD_SIZE, WRITE);
                break;
        case 8: accessDRAM(addr, (byte*)data, DOUBLEWORD_SIZE, WRITE);
                break;
        case 16:accessDRAM(addr, (byte*)data, QUADWORD_SIZE, WRITE);
                break;
        case 32:accessDRAM(addr, (byte*)data, OCTWORD_SIZE, WRITE);
        }
      // Write back miss OKAY
      } else {
        highlight_block(index, lruBlock);
        if (cache[index].block[lruBlock].dirty == DIRTY){
          unsigned int oldAddr = cache[index].block[lruBlock].tag << indexBits;
          oldAddr = oldAddr | index;
          oldAddr = oldAddr << (offsetBits);
          
          switch(block_size){
          case 4: accessDRAM(oldAddr, (byte*)cache[index].block[lruBlock].data, WORD_SIZE, WRITE);
                break;
          case 8: accessDRAM(oldAddr, (byte*)cache[index].block[lruBlock].data, DOUBLEWORD_SIZE, WRITE);
                break;
          case 16:accessDRAM(oldAddr, (byte*)cache[index].block[lruBlock].data, QUADWORD_SIZE, WRITE);
                break;
          case 32:accessDRAM(oldAddr, (byte*)cache[index].block[lruBlock].data, OCTWORD_SIZE, WRITE);
          }
        }
        
        switch(block_size){
        case 4: accessDRAM(addr-offset, (byte*)cache[index].block[lruBlock].data, WORD_SIZE, READ);
                break;
        case 8: accessDRAM(addr-offset, (byte*)cache[index].block[lruBlock].data, DOUBLEWORD_SIZE, READ);
                break;
        case 16:accessDRAM(addr-offset, (byte*)cache[index].block[lruBlock].data, QUADWORD_SIZE, READ);
                break;
        case 32:accessDRAM(addr-offset, (byte*)cache[index].block[lruBlock].data, OCTWORD_SIZE, READ);
      }

      memcpy(cache[index].block[lruBlock].data+offset, data, block_size);
      cache[index].block[lruBlock].dirty = DIRTY;
      cache[index].block[lruBlock].tag = tag;
      cache[index].block[lruBlock].valid = VALID;

      for (int i = 0; i < assoc; i++){
        if (cache[index].block[i].lru.value == 1){
          cache[index].block[i].lru.value = 0;
        }
      }
      cache[index].block[lruBlock].lru.value = 1;

    }
    // Cache write hit OKAY
    } else {
      highlight_offset(index, lruBlock, offset, HIT);
      memcpy(cache[index].block[lruBlock].data+offset, data, 4);

      // Write through hit OKAY
      if (memory_sync_policy == WRITE_THROUGH){
        switch(block_size){
        case 4: accessDRAM(addr, (byte*)data, WORD_SIZE, WRITE);
                //accessDRAM(addr, (byte*)cache[index].block[lruBlock].data, WORD_SIZE, WRITE);
                break;
        case 8: accessDRAM(addr, (byte*)data, DOUBLEWORD_SIZE, WRITE);
                break;
        case 16:accessDRAM(addr, (byte*)data, QUADWORD_SIZE, WRITE);
                break;
        case 32:accessDRAM(addr, (byte*)data, OCTWORD_SIZE, WRITE);
        }
      // Write back hit OKAY
      } else {
        cache[index].block[lruBlock].dirty = DIRTY;
      }

    }
  }
}
