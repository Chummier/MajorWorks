using System;
using System.IO;
using System.Collections.Generic;
using System.Runtime.Serialization.Formatters.Binary;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.InputSystem;

[System.Serializable]
public struct RaceHighscore
{
    public string name;
    public float time;

    public RaceHighscore(string username, float seconds)
    {
        name = username;
        time = seconds;
    }
    
}
[System.Serializable]
public struct RaceHighscoreList
{
    public List<RaceHighscore> list;

    public RaceHighscoreList(int length)
    {
        list = new List<RaceHighscore>(length);
        for (int i = 0; i < length; i++)
        {
            list.Add(new RaceHighscore("None", 00.00f));
        }
    }
}
[System.Serializable]
public struct CoinHighscore
{
    public string name;
    public int coins;

    public CoinHighscore(string username, int amount)
    {
        name = username;
        coins = amount;
    }
}
[System.Serializable]
public struct CoinHighscoreList
{
    public List<CoinHighscore> list;

    public CoinHighscoreList(int length)
    {
        list = new List<CoinHighscore>(length);
        for (int i = 0; i < length; i++)
        {
            list.Add(new CoinHighscore("None", 0));
        }
    }
}

[System.Serializable]
public class Data
{
    public Dictionary<int, CoinHighscoreList> localCoin;
    public Dictionary<int, RaceHighscoreList> localRace;
    public Dictionary<int, CoinHighscoreList> devCoin;
    public Dictionary<int, RaceHighscoreList> devRace;

    public Data(Dictionary<int, CoinHighscoreList> coinDict, Dictionary<int, RaceHighscoreList> raceDict)
    {
        localCoin = coinDict;
        localRace = raceDict;
    }

    public Data(Dictionary<int, CoinHighscoreList> coinDict, Dictionary<int, RaceHighscoreList> raceDict, Dictionary<int, CoinHighscoreList> coinDev, Dictionary<int, RaceHighscoreList> raceDev)
    {
        localCoin = coinDict;
        localRace = raceDict;
        devCoin = coinDev;
        devRace = raceDev;
    }
}

public class HighscoreSystem : MonoBehaviour
{
    private static int scoresAmount = 10;
    private GameObject[] CoinScoreSlots = new GameObject[scoresAmount];
    private GameObject[] CoinDevScoreSlots = new GameObject[scoresAmount / 2];

    private GameObject[] RaceScoreSlots = new GameObject[scoresAmount];
    private GameObject[] RaceDevScoreSlots = new GameObject[scoresAmount / 2];

    public Dictionary<int, CoinHighscoreList> CoinLevels;
    public Dictionary<int, RaceHighscoreList> RaceLevels;
    public Dictionary<int, CoinHighscoreList> DevCoinLevels;
    public Dictionary<int, RaceHighscoreList> DevRaceLevels;

    public Animator KeysAnimator;

    private float distanceToTV;
    private bool atTV;

    public GameObject player;
    public GameObject playerHUD;
    public GameObject highscoreMenu;

    private float currentLevel;
    public Text levelIndicator;

    // Start is called before the first frame update
    void Start()
    {
        atTV = false;

        currentLevel = 0;
        levelIndicator.text = "Level " + (currentLevel + 1).ToString();

        CoinLevels = new Dictionary<int, CoinHighscoreList>();
        RaceLevels = new Dictionary<int, RaceHighscoreList>();
        DevCoinLevels = new Dictionary<int, CoinHighscoreList>();
        DevRaceLevels = new Dictionary<int, RaceHighscoreList>();

        for (int i = 1; i < 6; i++)
        {
            CoinLevels.Add(i, new CoinHighscoreList(scoresAmount));
            RaceLevels.Add(i, new RaceHighscoreList(scoresAmount));
            DevCoinLevels.Add(i, new CoinHighscoreList(scoresAmount / 2));
            DevRaceLevels.Add(i, new RaceHighscoreList(scoresAmount / 2));
        }

        try
        {
            for (int i = 0; i < CoinScoreSlots.Length; i++)
            {
                CoinScoreSlots[i] = GameObject.Find("CoinNameScore" + (i + 1).ToString());
            }

            for (int i = 0; i < RaceScoreSlots.Length; i++)
            {
                RaceScoreSlots[i] = GameObject.Find("RaceNameScore" + (i + 1).ToString());
            }

            for (int i = 0; i < CoinDevScoreSlots.Length; i++)
            {
                CoinDevScoreSlots[i] = GameObject.Find("CoinDevScore" + (i + 1).ToString());
            }

            for (int i = 0; i < RaceDevScoreSlots.Length; i++)
            {
                RaceDevScoreSlots[i] = GameObject.Find("RaceDevScore" + (i + 1).ToString());
            }

            initializeDevScores();
            loadLocalData();
            changeLevel(0);

        }
        catch (Exception e)
        {
            Debug.Log("Dang, error in HighscoreSystem:\n" + e);
        }
    }

    private void Update()
    {
        if (atTV) {
            distanceToTV = (highscoreMenu.transform.position - player.transform.position).magnitude;
            if (distanceToTV >= 3)
            {
                atTV = false;
                playerHUD.SetActive(true);
            }
        }
    }

    private void changeLevel(int level)
    {

        for (int i = 0; i < CoinScoreSlots.Length; i++)
        {
            CoinScoreSlots[i].GetComponent<Text>().text = CoinLevels[level+1].list[i].name + " " + CoinLevels[level+1].list[i].coins.ToString();
        }
        for (int i = 0; i < CoinDevScoreSlots.Length; i++)
        {
            CoinDevScoreSlots[i].GetComponent<Text>().text = DevCoinLevels[level + 1].list[i].name + " " + DevCoinLevels[level + 1].list[i].coins.ToString();   
        }

        for (int i = 0; i < RaceScoreSlots.Length; i++)
        {
            RaceScoreSlots[i].GetComponent<Text>().text = RaceLevels[level+1].list[i].name + " " + RaceLevels[level+1].list[i].time.ToString("F");
        }
        for (int i = 0; i < RaceDevScoreSlots.Length; i++)
        {
            RaceDevScoreSlots[i].GetComponent<Text>().text = DevRaceLevels[level + 1].list[i].name + " " + DevRaceLevels[level + 1].list[i].time.ToString("F");
        }
    }

    public void levelInput(InputAction.CallbackContext context)
    {
        if (atTV)
        {
            // Next level
            if (context.ReadValue<Vector2>().x >= 1)
            {
                currentLevel = currentLevel + 0.5f; // Button gets pressed twice for some reason
                if (currentLevel >= 5)
                {
                    currentLevel = 0;
                }
                KeysAnimator.SetTrigger("RightTrigger");
            }

            // Previous level
            else if (context.ReadValue<Vector2>().x <= -1)
            {
                currentLevel = currentLevel - 0.5f; // Button gets pressed twice for some reason
                if (currentLevel <= -1)
                {
                    currentLevel = 4;
                }
                KeysAnimator.SetTrigger("LeftTrigger");
            }

            levelIndicator.text = "Level " + ((int)currentLevel + 1).ToString();
            changeLevel((int)currentLevel);
        }
    }

    public void updateRace(int level, string user, float value)
    {
        for (int i = 0; i < scoresAmount; i++)
        {
            if (value > RaceLevels[level].list[i].time)
            {
                for (int j = scoresAmount-1; j > i; j--)
                {
                    RaceLevels[level].list[j] = RaceLevels[level].list[j - 1];
                }
                RaceLevels[level].list[i] = new RaceHighscore(user, value);
                changeLevel(0);
                return;
            }
        }
    }

    private void updateDevRace(int level, string user, float value)
    {
        for (int i = 0; i < scoresAmount/2; i++)
        {
            if (value > DevRaceLevels[level].list[i].time)
            {
                for (int j = scoresAmount/2 - 1; j > i; j--)
                {
                    DevRaceLevels[level].list[j] = DevRaceLevels[level].list[j - 1];
                }
                DevRaceLevels[level].list[i] = new RaceHighscore(user, value);
                return;
            }
        }
    }

    public void updateCoin(int level, string user, int value)
    {
        for (int i = 0; i < scoresAmount; i++)
        {
            if (value > CoinLevels[level].list[i].coins)
            {
                for (int j = scoresAmount - 1; j > i; j--)
                {
                    CoinLevels[level].list[j] = CoinLevels[level].list[j - 1];
                }
                CoinLevels[level].list[i] = new CoinHighscore(user, value);
                changeLevel(0);
                return;
            }
        }
    }

    private void updateDevCoin(int level, string user, int value)
    {
        for (int i = 0; i < scoresAmount/2; i++)
        {
            if (value > DevCoinLevels[level].list[i].coins)
            {
                for (int j = scoresAmount/2 - 1; j > i; j--)
                {
                    DevCoinLevels[level].list[j] = DevCoinLevels[level].list[j - 1];
                }
                DevCoinLevels[level].list[i] = new CoinHighscore(user, value);
                return;
            }
        }
    }

    private void initializeDevScores()
    {
        updateDevRace(1, "Jared～", 19.64f);
        updateDevRace(1, "Jared～", 23.80f);
        updateDevRace(1, "Jared～", 14.42f);
        updateDevRace(1, "Jared～", 15.22f);
        updateDevRace(1, "Jared～", 9.62f);

        updateDevRace(2, "Jared～", 17.22f);
        updateDevRace(2, "Jared～", 12.86f);
        updateDevRace(2, "Jared～", 12.38f);
        updateDevRace(2, "Jared～", 17.60f);
        updateDevRace(2, "Jared～", 11.55f);

        updateDevRace(3, "Jared～", 12.34f);
        updateDevRace(3, "Jared～", 10.82f);
        updateDevRace(3, "Jared～", 9.94f);
        updateDevRace(3, "Jared～", 15.59f);
        updateDevRace(3, "Jared～", 18.55f);

        updateDevRace(4, "Jared～", 20.82f);
        updateDevRace(4, "Jared～", 23.74f);
        updateDevRace(4, "Jared～", 23.84f);
        updateDevRace(4, "Jared～", 22.61f);
        updateDevRace(4, "Jared～", 17.12f);

        updateDevRace(5, "Jared～", 1.64f);
        updateDevRace(5, "Jared～", 2.40f);
        updateDevRace(5, "Jared～", 2.48f);
        updateDevRace(5, "Jared～", 3.14f);
        updateDevRace(5, "Jared～", 2.30f);

        updateDevCoin(1, "Jared～", 7);
        updateDevCoin(1, "Jared～", 6);
        updateDevCoin(1, "Jared～", 9);
        updateDevCoin(1, "Jared～", 8);
        updateDevCoin(1, "Jared～", 5);

        updateDevCoin(2, "Jared～", 13);
        updateDevCoin(2, "Jared～", 10);
        updateDevCoin(2, "Jared～", 11);
        updateDevCoin(2, "Jared～", 9);
        updateDevCoin(2, "Jared～", 8);

        updateDevCoin(3, "Jared～", 5);
        updateDevCoin(3, "Jared～", 7);
        updateDevCoin(3, "Jared～", 6);
        updateDevCoin(3, "Jared～", 4);
        updateDevCoin(3, "Jared～", 9);

        updateDevCoin(4, "Jared～", 6);
        updateDevCoin(4, "Jared～", 7);
        updateDevCoin(4, "Jared～", 9);
        updateDevCoin(4, "Jared～", 5);
        updateDevCoin(4, "Jared～", 8);

        updateDevCoin(5, "Jared～", 8);
        updateDevCoin(5, "Jared～", 10);
        updateDevCoin(5, "Jared～", 13);
        updateDevCoin(5, "Jared～", 15);
        updateDevCoin(5, "Jared～", 11);
    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("Player"))
        {
            atTV = true;
            playerHUD.SetActive(false);
            player.GetComponent<PauseMenuScript>().togglePauseMenu(false);
            loadLocalData();
        }
    }

    public void saveLocalData()
    {
        string target = Application.persistentDataPath + "/scores.hs";
        BinaryFormatter formatter = new BinaryFormatter();
        FileStream file;

        if (File.Exists(target))
        {
            file = File.OpenWrite(target);
        } else
        {
            file = File.Create(target);
        }

        Data data = new Data(CoinLevels, RaceLevels);
        formatter.Serialize(file, data);
        file.Close();
    }

    public void loadLocalData()
    {
        string target = Application.persistentDataPath + "/scores.hs";
        BinaryFormatter formatter = new BinaryFormatter();
        FileStream file;

        if (File.Exists(target))
        {
            file = File.OpenRead(target);
        }
        else
        {
            Debug.Log("No highscores file found");
            return;
        }

        Data data = (Data)formatter.Deserialize(file);
        file.Close();

        CoinLevels = data.localCoin;
        RaceLevels = data.localRace;
    }
}
