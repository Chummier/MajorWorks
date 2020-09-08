using System;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.UI;
using UnityStandardAssets.Utility;
using UnityStandardAssets.Characters.FirstPerson;
using Random = UnityEngine.Random;

/* This script is from the same Unity game as the others
It's a script that comes with a standard set of assets for unity.
It handles 3D movement
I've omitted all the parts that I didn't write, and left notable enchancements I made
to the script

Enhancements like:
Double jumping

Slamming to the ground while mid-air

Having a sprint meter that appears while sprinting then disappears when walking
It detects when it's empty and then recharges

Having timed hints appear on screen: they show a progress bar below the text that 
shows when the hint will disappear

Allowing the player to climb ropes

Making the player's position match moving objects that they are standing on- this functionality
was not there by default

*/

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


   

    private void FixedUpdate()
    {
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

    private void OnControllerColliderHit(ControllerColliderHit hit)
    {

        // Make player move on moving objects
        if ((hit.gameObject.GetComponent<SlidingScript>() != null) && m_Input.x == 0 && m_Input.y == 0)
        {
            m_UseHeadBob = false;
            isSliding = true;
            this.transform.Translate(hit.gameObject.GetComponent<SlidingScript>().getVelocity() * Time.smoothDeltaTime, Space.World);
        }
        
        // Make player climb up ropes
        if (hit.gameObject.CompareTag("Rope"))
        {
            this.transform.Translate(new Vector3(0, Math.Abs(m_Input.y / 5), 0), Space.World);
            m_MoveDir.y = Math.Abs(m_Input.y / 5);
        }
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
}
