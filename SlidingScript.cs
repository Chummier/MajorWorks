﻿using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SlidingScript : MonoBehaviour {

	public float limit = 0;

	[Space]
	[Header("Movement vector has the form (vx, vy, vz)")]
	[Space]
	public float vx = 0;
	public float vy = 0;
	public float vz = 0;

	private Vector3 dV;
	private Vector3 startPos;
	private Vector3 newPos;

	// Use this for initialization
	void Start () {
		dV = new Vector3(vx,vy,vz);
		startPos = transform.position;
		newPos = transform.position;
	}

	public Vector3 getVelocity(){
		return dV;
	}

    // Update is called once per frame
    void FixedUpdate() {

        if ((newPos - startPos).magnitude > limit)
        {
            dV = -dV;
        }
        transform.Translate(dV * Time.smoothDeltaTime);
        newPos = transform.position;
	}
}
