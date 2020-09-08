using System;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.InputSystem;
using Random = UnityEngine.Random;

public class FreeroamSystem : MonoBehaviour {

	public GameObject freeroamLocation;
	private bool isFreeroam;

	public GameObject startLocation;
	public GameObject teleFlag;
	public GameObject teleLocation;

	public GameObject player;

    public GameObject progBarTitle;
    public Slider progressSlider;
    private bool progBarFlag;

    private GameObject[] rotators;
    private Quaternion[] rotations;

    private GameObject[] coins;
    private GameObject[] stopwatches;

    private bool isResetting;
    private bool isTeleporting;

    public AudioClip[] backgroundMusic;

    private float timeForNextMusic;

    // Use this for initialization
    void Start () {

        isFreeroam = false;
        progBarFlag = false;
        isResetting = false;
        isTeleporting = false;
        timeForNextMusic = 0;

        try
        {
            teleFlag.transform.position = startLocation.transform.position;
            teleFlag.SetActive(false);

            progBarTitle.GetComponent<Text>().text = "";
            progressSlider.gameObject.SetActive(false);
            progressSlider.value = 0;

            rotators = GameObject.FindGameObjectsWithTag("Rotating");
            rotations = new Quaternion[rotators.Length];

            for (int i = 0; i < rotators.Length; i++)
            {
                rotations[i] = rotators[i].transform.rotation;
            }

            coins = GameObject.FindGameObjectsWithTag("Coin");
            stopwatches = GameObject.FindGameObjectsWithTag("Stopwatch");

            for (int i = 0; i < stopwatches.Length; i++)
            {
                stopwatches[i].SetActive(false);
            }
        } catch (Exception e)
        {
            Debug.Log("Zoinks, FreeroamSystem had an error in Start():\n" + e);
        }
    }

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

    public void checkpointInput(InputAction.CallbackContext context)
    {
        if (isFreeroam)
        {
            placeCheckpoint();
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

	private void OnTriggerEnter(Collider c){
		
		if (c.gameObject.CompareTag ("Player")) {

            isFreeroam = true;

            try
            {
                teleFlag.SetActive(true);
                teleFlag.transform.position = freeroamLocation.transform.position;
                teleLocation.transform.position = freeroamLocation.transform.position;

                player.GetComponent<MyFPSController>().setDoubleJumpRecharge(true);
                player.GetComponent<MyFPSController>().rechargeToMaxSprint();

                player.GetComponent<MyFPSController>().toggleHint("Hold T to return to the bus", 10, true);
                player.GetComponent<MyFPSController>().playerHUD.SetActive(true);

                playMusic();

            } catch (Exception e)
            {
                Debug.Log("My my, an error in FreeroamSystem in OnTriggerEnter():\n" + e);
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

	public bool getFreeroamStatus(){
		return isFreeroam;
	}

    public void setIsFreeroam(bool flag)
    {
        isFreeroam = flag;
    }

    public void placeCheckpoint(){
        if (player.GetComponent<MyFPSController>().isGrounded() && isFreeroam)
        {
            teleLocation.transform.position = player.transform.position;
            teleFlag.transform.position = player.transform.position;
        }
	}

    public void resetMap(bool coinsStopwatches)
    {
        resetRotatingObjects();
        toggleCoinsStopwatches(coinsStopwatches);
    }

    public void teleport()
    {
        isFreeroam = false;

        toSpawn();
        resetMap(true);

        teleFlag.transform.position = startLocation.transform.position;
        teleFlag.SetActive(false);

        player.GetComponent<MyFPSController>().toggleHint("", 0, false);

        player.GetComponent<MyFPSController>().rechargeToMaxSprint();

        player.GetComponent<AudioSource>().Stop();
    }

    public void resetRotatingObjects()
    {
        for (int i = 0; i < rotators.Length; i++)
        {
            rotators[i].transform.rotation = rotations[i];
            rotators[i].GetComponent<Rigidbody>().angularVelocity = Vector3.zero;
        }
    }

    public void toggleCoinsStopwatches(bool flag)
    {
        player.GetComponent<MyFPSController>().coinCount = 0;
        player.GetComponent<MyFPSController>().updateCoinText(0);
        for (int i = 0; i < coins.Length; i++)
        {
            coins[i].SetActive(flag);
            stopwatches[i].SetActive(!flag);
        }
    }

    public void toSpawn()
    {
        player.transform.position = startLocation.transform.position;
        player.transform.rotation = startLocation.transform.rotation;
    }

    public void toFlag()
    {
        player.transform.position = teleLocation.transform.position;
        player.transform.rotation = teleLocation.transform.rotation;
    }
}