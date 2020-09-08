using System;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.InputSystem;
using Random = UnityEngine.Random;

/* This script is entirely written by me and is  part of a 3D platforming game in Unity
I've omitted a lot of the code except for notable components like:

A music system that selects a song randomly from a list, gets the length of that song,
and then stores the expected time that song will finish at. It then waits for that expected time
to play another random song

A system with progress bars, where while the player holds the Reset or Teleport buttons down it fills
a meter, and when it's full it resets or teleports. The buttons jam each other- teleport doesn't work if
reset is being held down and vice versa, and the progress bar resets after the button is let go

*/


    void FixedUpdate()
    {
        if (isFreeroam)
        {
            if (Time.unscaledTime >= timeForNextMusic)
            {
                playMusic();
            }
        }

        if (isResetting)
        {
            if (progressSlider.value == 100)
            {
                progressSlider.gameObject.SetActive(false);
                progBarTitle.GetComponent<Text>().text = "";
                progressSlider.value = 0;
                resetMap(true);
                isResetting = false;
            } else
            {
                progressSlider.gameObject.SetActive(true);
                progBarTitle.GetComponent<Text>().text = "Resetting Map...";
                progressSlider.value += 2;
            }
        }

        if (isTeleporting)
        {
            if (progressSlider.value == 100)
            {
                progressSlider.gameObject.SetActive(false);
                progBarTitle.GetComponent<Text>().text = "";
                progressSlider.value = 0;

                teleport();
                player.GetComponent<MyFPSController>().toggleHint("", 0, false);
                player.GetComponent<MyFPSController>().updateCoinText(0);
                isTeleporting = false;

            } else
            {
                progressSlider.gameObject.SetActive(true);
                progBarTitle.GetComponent<Text>().text = "Teleporting...";
                progressSlider.value += 2;
            }
        }
    }

    public void playMusic()
    {
        if (isFreeroam)
        {
            int index = (int)Random.Range(0, backgroundMusic.Length - 1);
            player.GetComponent<AudioSource>().PlayOneShot(backgroundMusic[index], PlayerPrefs.GetFloat("MusicVolume", 1f));

            timeForNextMusic = Time.time + backgroundMusic[index].length + 10f;
        }
    }
    
    public void teleportInput(InputAction.CallbackContext context)
    {
        if (isFreeroam && context.ReadValueAsButton() && !isResetting)
        {
            isTeleporting = true;
        } else
        {
            if (!isResetting)
            {
                progressSlider.gameObject.SetActive(false);
                progBarTitle.GetComponent<Text>().text = "";
                progressSlider.value = 0;
            }
            isTeleporting = false;
        }
    }
    
    public void resetInput(InputAction.CallbackContext context)
    {
        if (isFreeroam && context.ReadValueAsButton() && !isTeleporting)
        {
            isResetting = true;
        } else
        {
            if (!isTeleporting)
            {
                progressSlider.gameObject.SetActive(false);
                progBarTitle.GetComponent<Text>().text = "";
                progressSlider.value = 0;
            }
            isResetting = false;
        }
    }
