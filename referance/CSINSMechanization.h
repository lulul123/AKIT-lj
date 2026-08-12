/**
* @file
* @brief    IMU Mechanization
* @details  惯导机械编排 \n
* @author   Feng Zhu. Email: fzhu@whu.edu.cn
* @date     2018/01/17
* @version  2.3.20230417
* @par      Copyright(c) 2012-2021 School of Geodesy and Geomatics, University of Wuhan. All Rights Reserved.
*           POSMind Software Kernel, named IPS (Inertial Plus Sensing), this file is a part of IPS.
* @par      History:
*           2018/01/17,Feng Zhu, new \n
*/

/*
  ref: [1] 严恭敏,捷联惯导算法及车载组合导航系统研究[D],2004
       [2] 秦永元,惯性导航[M],2006
	   [3] Savage,Strapdown Inertial Navigation Integration Algorithm Design Part 1: Attitude Algorithms[J],1998
	   [4] Savage,Strapdown Inertial Navigation Integration Algorithm Design Part 2: Velocity and Position Algorithms[J],1998
	   [5] Savage,Strapdown analytics[M],2000
	   [6] Shin,Accuracy improvement of low cost INS/GPS for land applications[D],2001
	   [7] Titterton,Strapdown inertial navigation technology[M],1997
	   [8] Dambeck,A Single Frequency Strapdown Algorithm for Integrating IMUs in ECEF-Frame[J],2011
	   [9] Wu,Strapdown inertial navigation system algorithms based on dual quaternions[J],2005
	  [10] Wu,Strapdown Inertial Navigation System Algorithms Based on Geometric Algebra[J],2012
*/


#ifndef SINSTK_CSINSMECHANIZATION_H
#define SINSTK_CSINSMECHANIZATION_H

#include "BASEDLL/CEarthGravityModel.h"

#include "SINSDLL/SINSSDC.h"
#include "SINSDLL/CSINSFFStream.h"

using namespace std;
using namespace basetk;

namespace sinstk
{

	class IPS_DLLEXPORT CSINSMechanizationQuat
	{
		// Constructor and Destructor
	public:

		CSINSMechanizationQuat();

		virtual ~CSINSMechanizationQuat();

		// step 1 : 初始化
		bool init(NAVI_SYS naviSys, bool bReverse, bool bOnlyUpdateAtt, CIMUFFStream* pIMUFFStream);
		bool initInfo(double Rbn[9], double vel[3], double pos[3]);

	public:

		bool runOneMechanization(double Rbn[9], double vel[3], double pos[3]);
		void getAttitude(double att[3]);

		// 输出数据
		GPSTIME m_time;    ///< 当前机械编排结果的时间点
		double  m_Rbn[9];
		double  m_Qbn[4];
		double  m_Velocity[3];
		double  m_Position[3];
				
		bool m_bExport;
		bool m_Exp_bUpdate;
		GPSTIME m_Exp_UpdateTime;
		double m_Exp_UpdateDt;

		// 内部机械编排的观测值
		double m_DeltTime;
		double m_dAngle[3];
		double m_dVelocity[3];

		bool   m_bSFA;    ///< 是否为静态精对准
		bool    m_bGetData;

		// Implementation
	protected:

		bool getIMUDatas();
		bool getIMUDatasExport();
		bool PrepareUpdate(bool bpos = true, bool bvel = true);
		void runPositionUpdateForward();
		void runVelocityUpdateForward();
		void runAttitudeUpdateForward();
		void runPositionUpdateReverse();
		void runVelocityUpdateReverse();
		void runAttitudeUpdateReverse();

		// Members
	private:

		CIMUFFStream* m_pIMUFFStream;
		NAVI_SYS      m_NaviSys;        ///< 机械编排导航系(默认为L系)
		bool          m_bOnlyUpdateAtt; ///< 只做姿态更新,用于双天测姿融合 
		bool          m_bReverse;       ///< 逆向解算

		CSimpleEGM m_SimpleEMG;          ///< 重力场模型
		double m_Gravity[3];
		double m_Wnie[3];
		double m_Wnen[3];
		double m_Anin[3];
		double m_rmh;
		double m_rnh;

		bool m_bAttUPTFirst; ///< 是否为姿态先更新的机械编排算法,默认true

		// 速度更新中的临时变量
		double m_VelPre[3];
		double m_DeltVelGCor[3];
		double m_DeltVelSFm[3];
		double m_DeltVelSFmb[3];
		double m_DeltVelSFm_1[3];

		//GPSTIME m_timeFix;
		GPSTIME m_timePre;
		//double  m_DeltTimeFix;
		
		bool    m_bBack;
		//double  m_DeltTf;
		//double  m_DeltTb;
		
		// 类内部使用的临时变量
		double m_Q1[4];
		double m_Q2[4];
		double m_Q3[4];
		double m_M1[3];
		double m_M2[3];
		double m_M3[3];
		double m_R1[9];
	};
}




void TestMech();
void TestKinematicMech();

#endif