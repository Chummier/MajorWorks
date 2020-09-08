using System;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.UI;
using UnityStandardAssets.Utility;
using UnityStandardAssets.Characters.FirstPerson;
using Random = UnityEngine.Random;

[RequireComponent(typeof(CharacterController))]
[RequireComponent(typeof(AudioSource))]
public class MyFPSController : MonoBehaviour
{
    [SerializeField] private bool m_IsWalking;
    [SerializeField] private float m_WalkSpeed;
    [SerializeField] private float m_RunSpeed;
    [SerializeField] [Range(0f, 1f)] private float m_RunstepLenghten;
    [SerializeField] private float m_JumpSpeed;
    [SerializeField] private float m_StickToGroundForce;
    [SerializeField] public float m_GravityMultiplier;
    [SerializeField] public MouseLook m_MouseLook;
    [SerializeField] private bool m_UseFovKick;
    [SerializeField] private FOVKick m_FovKick = new FOVKick();
    [SerializeField] private bool m_UseHeadBob;
    [SerializeField] private CurveControlledBob m_HeadBob = new CurveControlledBob();
    [SerializeField] private LerpControlledBob m_JumpBob = new LerpControlledBob();
    [SerializeField] private float m_StepInterval;
    [SerializeField] private AudioClip[] m_FootstepSounds;    // an array of footstep sounds that will be randomly selected from.
    [SerializeField] private AudioClip m_JumpSound;           // the sound played when character leaves the ground.
    [SerializeField] private AudioClip m_LandSound;           // the sound played when character touches back on ground.
    [SerializeField] private AudioSource m_AudioSource;

    private Camera m_Camera;
    private bool m_Jump;
    private Vector2 m_Input;
    private Vector3 m_MoveDir = Vector3.zero;
    private CharacterController m_CharacterController;
    private CollisionFlags m_CollisionFlags;
    private bool m_PreviouslyGrounded;
    private Vector3 m_OriginalCameraPosition;
    private float m_StepCycle;
    private float m_NextStep;
    private bool m_Jumping;

    // Sprint Variables //
    public Slider sprintSlider;
    private float sprintLevel;
    private bool sprintCanRecharge;
    [SerializeField] private float maxSprint;
    [SerializeField] private float sprintDrain;
    //////////////////////

    private float initialGravityMultiplier;
    private bool isSlamming;
    public float defaultGravity;

    private bool isSliding;

    private bool doubleJumpRecharges;
    public Image doubleJumpIndicator;
    private int extraJumps;
    private int maxJumps;

    public Text coinCountText;
    public int coinCount;
    public GameObject plusOneCoin;

    public GameObject ChallengeTV;
    public GameObject playerHUD;

    public Slider hintSlider;
    public GameObject hint;
    public bool doHint;

    private bool showHints;
    private bool canSlam;

    public GameObject Fog;
    private bool coinsAddTime;

    private float musicVolume;
    private float seVolume;

    public bool inTutorial;

    [SerializeField] private AudioClip coinSound;
    [SerializeField] private AudioClip[] musicClips;
    [SerializeField] private AudioClip[] leafFootsteps;
    [SerializeField] private AudioClip leafJumpSound;
    [SerializeField] private AudioClip leafLandSound;

    private bool onLeaves;
    public int currentLevel = -1;

    private void Awake()
    {
        currentLevel = -1;
        inTutorial = false;

        try
        {
            playerHUD.SetActive(false);
        } catch (Exception e)
        {
            Debug.Log("Error in MyFPSController in Awake():\n" + e);
        }

    }

    // Use this for initialization
    private void Start()
    {
        try
        {
            m_CharacterController = GetComponent<CharacterController>();
            m_Camera = Camera.main;
            m_OriginalCameraPosition = m_Camera.transform.localPosition;
            m_FovKick.Setup(m_Camera);
            m_HeadBob.Setup(m_Camera, m_StepInterval);
            m_AudioSource = GetComponent<AudioSource>();
            m_MouseLook.Init(transform, m_Camera.transform);

            m_MouseLook.SetCursorLock(true);
            Cursor.visible = false;
            
            sprintSlider.value = maxSprint;
            sprintSlider.gameObject.SetActive(false);

            RenderSettings.skybox = (Material)Resources.Load("WispySkyboxMat2");
            coinCountText.GetComponent<Text>().text = "";

            doubleJumpIndicator.gameObject.SetActive(false);
            Application.targetFrameRate = 60;

            hint.SetActive(false);

            hintSlider.gameObject.SetActive(false);
            initialGravityMultiplier = m_GravityMultiplier;
            Fog.SetActive(false);

        } catch (Exception e)
        {
            Debug.Log("MyFPSController had an oopsie in Start():\n" + e);
        }

        m_StepCycle = 0f;
        m_NextStep = m_StepCycle / 2f;
        m_Jumping = false;

        m_IsWalking = true;

        isSliding = true;

        sprintLevel = maxSprint;
        sprintCanRecharge = true;

        doubleJumpRecharges = true;
        extraJumps = 1;
        maxJumps = 1;

        coinCount = 0;
        doHint = false;

        isSlamming = false;
        defaultGravity = 2;

        if (PlayerPrefs.GetInt("Hints", 1) == 1)
        {
            showHints = true;
        } else
        {
            showHints = false;
        }

        canSlam = false;
        coinsAddTime = false;

        musicVolume = PlayerPrefs.GetFloat("MusicVolume", 1f);
        seVolume = PlayerPrefs.GetFloat("SEVolume", 1f);

        onLeaves = false;
    }

    // Update is called once per frame
    private void Update()
    {
        RotateView();

        if (!m_PreviouslyGrounded && m_CharacterController.isGrounded)  // If the player just completed a jump
        {
            StartCoroutine(m_JumpBob.DoBobCycle());     // Play jump landing sequence
            PlayLandingSound();
            m_MoveDir.y = 0f;
            m_Jumping = false;
            if (doubleJumpRecharges)
            {
                extraJumps = maxJumps;
            }
        }
        if (!m_CharacterController.isGrounded && !m_Jumping && m_PreviouslyGrounded)    // If the player was just falling
        {
            m_MoveDir.y = 0f;   // Kill movement speed upon landing without jump landing sequence
        }

        m_PreviouslyGrounded = m_CharacterController.isGrounded; // Was the player grounded the previous frame?
    }

    public void jumpInput(InputAction.CallbackContext context)
    {
        if (m_CharacterController.isGrounded && context.ReadValueAsButton())
        {
            m_Jump = true;
        }
        else if (extraJumps > 0 && context.ReadValueAsButton())
        {
            m_Jump = true;
            if (--extraJumps == 0)
            {
                updateJumpIndicator(false);
            }
        }
    }

    public void slamInput(InputAction.CallbackContext context)
    {
        if (canSlam)
        {
            isSlamming = true;
        }
    }

    public void sprintInput(InputAction.CallbackContext context)
    {
        // This results in speed = m_RunSpeed in FixedUpdate
        if (context.ReadValueAsButton() && sprintLevel > 0 && !isSliding)
        {
            m_IsWalking = false;
            sprintSlider.gameObject.SetActive(true);
        }
        else
        {
            m_IsWalking = true;
            sprintSlider.gameObject.SetActive(false);
        }
    }

    public void movementInput(InputAction.CallbackContext context)
    {
        // Read input
        m_Input = context.ReadValue<Vector2>();

        // normalize input if it exceeds 1 in combined length:
        if (m_Input.sqrMagnitude > 1)
        {
            m_Input.Normalize();
        }
    }

    public void lookInput(InputAction.CallbackContext context)
    {
        Vector2 rotation = context.ReadValue<Vector2>();

        m_Camera.transform.localRotation = Quaternion.Euler(rotation.x, rotation.y, 0);
    }

    private void PlayLandingSound()
    {
        if (onLeaves)
        {
            m_AudioSource.clip = leafLandSound;
        } else
        {
            m_AudioSource.clip = m_LandSound;
        }
        m_AudioSource.PlayOneShot(m_AudioSource.clip, seVolume);
        m_NextStep = m_StepCycle + .5f;
    }


    private void FixedUpdate()
    {
        Cursor.visible = false;

        bool waswalking = m_IsWalking;

        // set the desired speed to be walking or running
        float speed;

        speed = m_IsWalking ? m_WalkSpeed : m_RunSpeed;

        // handle speed change to give an fov kick
        // only if the player is going to a run, is running and the fovkick is to be used
        if (m_IsWalking != waswalking && m_UseFovKick && m_CharacterController.velocity.sqrMagnitude > 0)
        {
            StopAllCoroutines();
            StartCoroutine(!m_IsWalking ? m_FovKick.FOVKickUp() : m_FovKick.FOVKickDown());
        }

        // always move along the camera forward as it is the direction that it being aimed at
        Vector3 desiredMove = transform.forward * m_Input.y + transform.right * m_Input.x;

        // get a normal for the surface that is being touched to move along it
        RaycastHit hitInfo;
        Physics.SphereCast(transform.position, m_CharacterController.radius, Vector3.down, out hitInfo,
                           m_CharacterController.height / 2f, Physics.AllLayers, QueryTriggerInteraction.Ignore);
        desiredMove = Vector3.ProjectOnPlane(desiredMove, hitInfo.normal).normalized;

        m_MoveDir.x = desiredMove.x * speed;
        m_MoveDir.z = desiredMove.z * speed;


        if (m_CharacterController.isGrounded) // If the player is on the ground
        {
            m_MoveDir.y = -m_StickToGroundForce; // Add gravity as a baseline for falling
        }
        else
        {
            m_MoveDir += Physics.gravity * m_GravityMultiplier * Time.fixedDeltaTime; // Accelerate the player to the ground
        }

        if (m_Jump) // If the player pressed Jump
        {
            m_MoveDir.y = m_JumpSpeed;  // Do jump sequence
            PlayJumpSound();
            m_Jump = false;
            m_Jumping = true;
        }

        m_CollisionFlags = m_CharacterController.Move(m_MoveDir * Time.fixedDeltaTime);

        ProgressStepCycle(speed);
        UpdateCameraPosition(speed);

        m_MouseLook.UpdateCursorLock();

        // Sprint Mechanics //
        if (speed == m_RunSpeed && !isSliding)
        {
            sprintLevel -= sprintDrain * Time.smoothDeltaTime;
            sprintSlider.value = sprintLevel;
        }
        if (sprintLevel <= 0 && sprintCanRecharge)
        {
            sprintSlider.gameObject.SetActive(true);
            m_IsWalking = true;
            sprintSlider.value += sprintDrain / 2 * Time.smoothDeltaTime;

            if (sprintSlider.value >= maxSprint)
            {
                sprintLevel = maxSprint;
                sprintSlider.gameObject.SetActive(false);
            }
        }
        //////////////////////

        if (doHint)
        {
            hintSlider.value -= Time.fixedDeltaTime;
            if (hintSlider.value <= 0f)
            {
                toggleHint("", 0, false);
            }
        }

        if (isSlamming)
        {
            if (!m_CharacterController.isGrounded)
            {
                m_GravityMultiplier = 5;
            } else
            {
                m_GravityMultiplier = initialGravityMultiplier;
                isSlamming = false;
            }

        }
    }

    private void PlayJumpSound()
    {
        if (onLeaves)
        {
            m_AudioSource.clip = leafJumpSound;
        } else
        {
            m_AudioSource.clip = m_JumpSound;
        }
        m_AudioSource.PlayOneShot(m_AudioSource.clip, seVolume);
    }


    private void ProgressStepCycle(float speed)
    {
        if (m_CharacterController.velocity.sqrMagnitude > 0 && (m_Input.x != 0 || m_Input.y != 0))
        {
            m_StepCycle += (m_CharacterController.velocity.magnitude + (speed * (m_IsWalking ? 1f : m_RunstepLenghten))) *
                         Time.fixedDeltaTime;
        }

        if (!(m_StepCycle > m_NextStep))
        {
            return;
        }

        m_NextStep = m_StepCycle + m_StepInterval;

        PlayFootStepAudio();
    }


    private void PlayFootStepAudio()
    {
        if (!m_CharacterController.isGrounded)
        {
            return;
        }
        // pick & play a random footstep sound from the array,
        // excluding sound at index 0

        int n;

        if (onLeaves)
        {
            n = Random.Range(1, leafFootsteps.Length);
            m_AudioSource.clip = leafFootsteps[n];
            leafFootsteps[n] = leafFootsteps[0];
            leafFootsteps[0] = m_AudioSource.clip;
        } else
        {
            n = Random.Range(1, m_FootstepSounds.Length);
            m_AudioSource.clip = m_FootstepSounds[n];
            m_FootstepSounds[n] = m_FootstepSounds[0];
            m_FootstepSounds[0] = m_AudioSource.clip;
        }

        m_AudioSource.PlayOneShot(m_AudioSource.clip, seVolume);
        // move picked sound to index 0 so it's not picked next time
    }


    private void UpdateCameraPosition(float speed)
    {
        Vector3 newCameraPosition;
        if (!m_UseHeadBob)
        {
            return;
        }
        if (m_CharacterController.velocity.magnitude > 0 && m_CharacterController.isGrounded)
        {
            m_Camera.transform.localPosition =
                m_HeadBob.DoHeadBob(m_CharacterController.velocity.magnitude +
                                  (speed * (m_IsWalking ? 1f : m_RunstepLenghten)));
            newCameraPosition = m_Camera.transform.localPosition;
            newCameraPosition.y = m_Camera.transform.localPosition.y - m_JumpBob.Offset();
        }
        else
        {
            newCameraPosition = m_Camera.transform.localPosition;
            newCameraPosition.y = m_OriginalCameraPosition.y - m_JumpBob.Offset();
        }
        m_Camera.transform.localPosition = newCameraPosition;
    }

    private void RotateView()
    {
        m_MouseLook.LookRotation(transform, m_Camera.transform);
    }

    private void OnTriggerEnter(Collider c)
    {
        if (c.gameObject.CompareTag("Coin"))
        {
            c.gameObject.SetActive(false);
            playCoinSound();
            updateCoinText(++coinCount);
            if (coinsAddTime)
            {
                ChallengeTV.GetComponent<ChallengeSystem>().addToTimer(5);
                plusOneCoin.GetComponent<TextMesh>().text = "+5 SECONDS";
            } else
            {
                plusOneCoin.GetComponent<TextMesh>().text = "+1 COIN";
            }
            plusOneCoin.GetComponent<Animator>().SetTrigger("FlashPlusOne");
        }

        if (c.gameObject.CompareTag("Stopwatch"))
        {
            c.gameObject.SetActive(false);
            playCoinSound();
            try { 
                ChallengeTV.GetComponent<ChallengeSystem>().addToTimer(5);
            } catch (Exception e)
            {
                Debug.Log("Collected a stopwatch in the tutorial:\n" + e);
            }
            plusOneCoin.GetComponent<TextMesh>().text = "+5 SECONDS";
            plusOneCoin.GetComponent<Animator>().SetTrigger("FlashPlusOne");
        }
    }

    private void OnControllerColliderHit(ControllerColliderHit hit)
    {

        // Make player move on moving objects
        if ((hit.gameObject.GetComponent<SlidingScript>() != null) && m_Input.x == 0 && m_Input.y == 0)
        {
            m_UseHeadBob = false;
            isSliding = true;
            this.transform.Translate(hit.gameObject.GetComponent<SlidingScript>().getVelocity() * Time.smoothDeltaTime, Space.World);
        }
        else
        {
            m_UseHeadBob = true;
            isSliding = false;
        }

        if (hit.gameObject.CompareTag("Leaves"))
        {
            onLeaves = true;
            
        } else
        {
            onLeaves = false;
        }

        // Make player climb up ropes
        if (hit.gameObject.CompareTag("Rope"))
        {
            this.transform.Translate(new Vector3(0, Math.Abs(m_Input.y / 5), 0), Space.World);
            m_MoveDir.y = Math.Abs(m_Input.y / 5);
        }

        // Activate disappearing blocks
        if (hit.gameObject.CompareTag("Disappearing") && hit.gameObject.GetComponent<AllowDisappearing>().getFlag())
        {
            hit.gameObject.GetComponent<AllowDisappearing>().toggle();
            hit.gameObject.GetComponent<Animator>().SetTrigger("Disappear");
        }

        Rigidbody body = hit.collider.attachedRigidbody;
        //dont move the rigidbody if the character is on top of it
        if (m_CollisionFlags == CollisionFlags.Below)
        {
            return;
        }

        if (body == null || body.isKinematic)
        {
            return;
        }
        body.AddForceAtPosition(m_CharacterController.velocity * 0.1f, hit.point, ForceMode.Impulse);
    }

    public void setDoubleJumpRecharge(bool flag)
    {
        doubleJumpRecharges = flag;
    }

    public void setExtraJumps(int amount)
    {
        extraJumps = amount;
        maxJumps = amount;
    }

    public void setSprintCanRecharge(bool flag)
    {
        sprintCanRecharge = flag;
    }

    public void rechargeToMaxSprint()
    {
        sprintLevel = maxSprint;
        sprintSlider.value = maxSprint;
        sprintSlider.gameObject.SetActive(false);
    }

    private void playCoinSound()
    {
        m_AudioSource.PlayOneShot(coinSound, seVolume);
    }

    public void playMusic(int level)
    {
        m_AudioSource.PlayOneShot(musicClips[level], musicVolume);
        currentLevel = level;
    }

    public void stopMusic()
    {
        currentLevel = -1;
        m_AudioSource.Stop();
    }

    public void setVolumes(float se, float mus)
    {
        setSEVolume(se);
        setMusicVolume(mus);
    }

    public void setSEVolume(float vol)
    {
        seVolume = vol;
        PlayerPrefs.SetFloat("SEVolume", vol);
    }

    public void setMusicVolume(float vol)
    {
        musicVolume = vol;
        PlayerPrefs.SetFloat("MusicVolume", vol);
        m_AudioSource.Stop();
        if (currentLevel != -1)
        {
            playMusic(currentLevel);
        }
    }

    public void updateCoinText(int amount)
    {
        if (amount == 1)
        {
            coinCountText.GetComponent<Text>().text = "1 COIN";
        }
        else if (amount == 0)
        {
            coinCountText.GetComponent<Text>().text = "";
        }
        else
        {
            coinCountText.GetComponent<Text>().text = amount.ToString() + " COINS";
        }
    }

    public void updateJumpIndicator(bool hasDouble)
    {
        if (hasDouble)
        {
            doubleJumpIndicator.color = Color.green;
        }
        else
        {
            doubleJumpIndicator.color = Color.red;
        }
    }

    public bool isGrounded()
    {
        return m_CharacterController.isGrounded;
    }

    public void toggleHint(string blessing, float seconds, bool show)
    {
        if (showHints)
        {
            hint.SetActive(show);
            hintSlider.gameObject.SetActive(show);
            hintSlider.maxValue = seconds;
            hintSlider.value = seconds;
            hint.GetComponent<Text>().text = blessing;
            doHint = show;
        }
    }

    public void setGravity(float multiplier)
    {
        initialGravityMultiplier = multiplier;
        m_GravityMultiplier = multiplier;
    }

    public void resetGravity()
    {
        initialGravityMultiplier = defaultGravity;
        m_GravityMultiplier = defaultGravity;
    }

    public void setHeadBob(float val)
    {
        m_HeadBob.HorizontalBobRange = val;
        m_HeadBob.VerticalBobRange = val;
        PlayerPrefs.SetFloat("Bobbing", val);
    }

    public void setShowHints(bool flag)
    {
        if (flag)
        {
            PlayerPrefs.SetInt("Hints", 1);
        } else
        {
            PlayerPrefs.SetInt("Hints", 0);
        }
        
        showHints = flag;
    }

    public void updateCanSlam(bool flag)
    {
        canSlam = flag;
    }

    public void updateCoinsAddTime(bool flag)
    {
        coinsAddTime = flag;
    }
}
