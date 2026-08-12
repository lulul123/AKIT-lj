#include "SINSDLL/CLooseCoupling.h"

#include "BASEDLL/BaseApp.h"
#include "BASEDLL/BaseTime.h"
#include "BASEDLL/CPose3D.h"
#include "BASEDLL/BaseMath.h"
#include "BASEDLL/BaseCmnFunc.h"

#include "SINSDLL/CSINSAlignment.h"
#include "SINSDLL/SINSCmnFunc.h"

using namespace basetk;

namespace sinstk
{








	// ************************************************************************************************************************************** //
	//  previous class :  ;     next class : CLooseCouplingBase                                                                               //
	// ************************************************************************************************************************************** //









	CLooseCouplingBase::CLooseCouplingBase()
	{
		m_pSINSOpt			= NULL;
		m_pSINSSol			= NULL;
		m_pIMUFFStream		= NULL;
		m_pELCFFStream		= NULL;
		m_ProcFlag			= 0;
		m_ProcInfo			= "";
		m_bInitLC			= false;
		m_bReverse			= false;
		m_UpdateDeltT		= 0.5;
		m_bStateUpdate		= false;
		m_bELCData			= false;
		m_MeasDeltT			= 1.0;
		m_bSTC				= false;
		m_bStateUpdate_STC	= false;
		m_bELCData_STC		= false;
		m_nZPT              = 0;
	}


	CLooseCouplingBase::~CLooseCouplingBase()
	{
		m_GLSInfo.free();

		m_pSINSOpt		= NULL;
		m_pSINSSol		= NULL;
		m_pIMUFFStream	= NULL;
		m_pELCFFStream	= NULL;
	}


	bool CLooseCouplingBase::initLCBase()
	{
		if (m_pSINSOpt     == NULL) return false;
		if (m_pSINSSol     == NULL) return false;
		if (m_pIMUFFStream == NULL) return false;
		if (m_pELCFFStream == NULL) return false;

		bool bTimeFlag = true;
		if (m_pELCFFStream && m_pELCFFStream->m_time.GPSWeek <= 10 || m_pIMUFFStream->m_time.GPSWeek <= 10) bTimeFlag = false;
		if (m_pSINSOpt->m_IMUBeginTime.GPSWeek <= 10 || m_pSINSOpt->m_IMUEndTime.GPSWeek <= 10) bTimeFlag = false;
		if (bTimeFlag == false)
		{
			string msg("Fail to obtain initial time.");
			DumpException(msg, LOGT_ERROR | LOGT_AFX, DUMP_LOCATION);
		}

		// 当前初始时间
		if (m_pELCFFStream)
		{
			m_UpdateTime = m_pELCFFStream->m_time;
			m_MeasDeltT = fabs(m_pELCFFStream->m_nominalDeltT);
		}
		else
		{
			if(m_bReverse) m_UpdateTime = m_pSINSOpt->m_IMUEndTime;
			else           m_UpdateTime = m_pSINSOpt->m_IMUBeginTime;
			m_MeasDeltT = 1.0;
		}
		

		m_UpdateDeltT = fabs(m_pSINSOpt->m_StateUPTDeltT);
		if (m_MeasDeltT <= 0.0) return false;

		// SINS状态更新时间间隔,状态更新时间是观测更新的一半
		//if (m_UpdateDeltT <= 0 || m_UpdateDeltT >= m_MeasDeltT) m_UpdateDeltT = m_MeasDeltT / 2.0;

		m_StateTime = GPSTIME(m_pIMUFFStream->m_time.GPSWeek, floor(_TOW(m_pIMUFFStream->m_time))); // 取整的时间
		
		if (m_bReverse)
		{
			//m_StateTime += 1.0;
			m_StateTime += 2 * m_UpdateDeltT;
			while (m_StateTime > m_pIMUFFStream->m_time) m_StateTime -= m_UpdateDeltT;
		}
		else
		{
			while (m_StateTime < m_pIMUFFStream->m_time) m_StateTime += m_UpdateDeltT;
		}

		for (int i = 0; i < 3; i++)
		{
			if (m_bReverse)
			{
				m_GLSInfo.Position[i] = m_pSINSOpt->m_InitPosb[i];
				m_GLSInfo.Velocity[i] = m_pSINSOpt->m_InitVelb[i];
				m_GLSInfo.Attitude[i] = m_pSINSOpt->m_InitAttb[i];
				m_GLSInfo.GyrosBias[i] = m_pSINSOpt->m_InitGyrosbiasb[i];
				m_GLSInfo.AccelBias[i] = m_pSINSOpt->m_InitAccelbiasb[i];
			}
			else
			{
				m_GLSInfo.Position[i] = m_pSINSOpt->m_InitPosf[i];
				m_GLSInfo.Velocity[i] = m_pSINSOpt->m_InitVelf[i];
				m_GLSInfo.Attitude[i] = m_pSINSOpt->m_InitAttf[i];
				m_GLSInfo.GyrosBias[i] = m_pSINSOpt->m_InitGyrosbiasf[i];
				m_GLSInfo.AccelBias[i] = m_pSINSOpt->m_InitAccelbiasf[i];
			}
		}
		
		if (m_pSINSOpt->m_bAzimuth) Azimuth2Attitude(m_GLSInfo.Attitude, m_GLSInfo.Attitude);
		
		if(m_pSINSOpt->m_NaviSys == NAVI_E)
		{
			double LLH[3];
			XYZ2LLH(m_GLSInfo.Position,LLH);
			Attitude2Rbe(m_GLSInfo.Attitude,LLH, m_GLSInfo.Rbn);
		}
		else if(m_pSINSOpt->m_NaviSys == NAVI_L)
		{
			double Pos[3]; M31EQU(m_GLSInfo.Position,Pos);
			XYZ2LLH(Pos, m_GLSInfo.Position);
			Vxyz2enu(m_GLSInfo.Position, m_GLSInfo.Velocity, m_GLSInfo.Velocity);
			Attitude2Rbl(m_GLSInfo.Attitude, m_GLSInfo.Rbn);
		}

		// 初始化机械编排
		if (m_SINSMech.init(m_pSINSOpt->m_NaviSys, m_bReverse, false, m_pIMUFFStream) == false) return false;
				
		return initKF();
	}


	bool CLooseCouplingBase::initKF()
	{
		InitSINS_StateIndex(m_pSINSOpt, &m_GLSInfo, 0, 0, m_bReverse);
		setSINSKF_InitialState(m_pSINSOpt, &m_GLSInfo);
		setSINSKF_Qc_Const(m_pSINSOpt, &m_GLSInfo);
		setSINSKF_CF_Const(m_bReverse, m_pSINSOpt, &m_GLSInfo);
		return true;
	}


	bool CLooseCouplingBase::KFPre()
	{			
		// 1. 进行机械编排
		// lj毕设：(m_pIMUFFStream->m_bAccel  m_pIMUFFStream->m_bGyros     在这个地方输出惯导原始数据
		//double       m_DeltVelocity[3];  ///< 速度增量(m/s)
		//double       m_DeltAngle[3];     ///< 角度增量(rad)
		//double       m_bAccel[3];        ///< b系比力
		//double       m_bGyros[3];        ///< b系角速度
		// lj毕设：m_GLSInfo.Velocity
		if (m_SINSMech.runOneMechanization(m_GLSInfo.Rbn, m_GLSInfo.Velocity, m_GLSInfo.Position) == false) { m_ProcFlag = 3; return false; }

		//输出原始IMU数据（机械编排后）
		//static FILE* fp1 = NULL;
		//if (!fp1)
		//{
		//	fp1 = fopen("Vel.txt", "w");
		//	if (fp1)
		//		fprintf(fp1, "Time Vx Vy Vz\n");
		//}

		//if (fp1)
		//{
		//	fprintf(fp1,
		//		"%.6f %.6f %.6f %.6f \n",
		//		m_UpdateTime.secsOfWeek+ m_UpdateTime.fracOfSec,
		//		m_GLSInfo.Velocity[0], m_GLSInfo.Velocity[1], m_GLSInfo.Velocity[2]
		//	);
		//	fflush(fp1);   // 关键：强制写入磁盘
		//}


		// 计算RTS增益
		if (m_GLSInfo.RTS_bStart)
		{
			m_GLSInfo.RTS_Ck = m_GLSInfo.StateP;
			m_GLSInfo.RTS_bStart = false;
		}
		
		// 这句话当时写着为了啥??
		//if (m_bStateUpdate)
		//{
		//	M31EQU(m_pIMUFFStream->m_bAccel, m_GLSInfo.bAccelCF);
		//	M31EQU(m_pIMUFFStream->m_bGyros, m_GLSInfo.bGyrosCF);
		//}

		if (MatrixNorm2(3, 1, m_GLSInfo.bAccelCF) <= 0.0) M31EQU(m_pIMUFFStream->m_bAccel, m_GLSInfo.bAccelCF);
		if (MatrixNorm2(3, 1, m_GLSInfo.bGyrosCF) <= 0.0) M31EQU(m_pIMUFFStream->m_bGyros, m_GLSInfo.bGyrosCF);

		// 2. 状态更新
		setSINSKF_EarthPara(m_pSINSOpt, &m_GLSInfo, &m_SimpleEGM, m_GLSInfo.Position, m_GLSInfo.Velocity, true, true);
		setSINSKF_Qc_Change(m_pSINSOpt, &m_GLSInfo, m_pIMUFFStream->m_bIn);
		setSINSKF_CF_Change(m_bReverse, m_pSINSOpt, &m_GLSInfo);
		//lj毕设：KFPrediction可以用来输出状态转移矩阵
	    //lj毕设：m_GLSInfo.StateX;m_GLSInfo.StateP, m_GLSInfo.SINSPhi（前一时刻的P矩阵、状态转移矩阵）
		// 本来在这里输出状态
		//static FILE* fp3 = fopen("StateP.txt", "w");
		//if (fp3&&(m_UpdateTime.fracOfSec<=0.00)) {
		//	static int header_written = 0;
		//	if (!header_written) {
		//		fprintf(fp3, "WeekSecond StateP\n");
		//		header_written = 1;
		//	}
		//	fprintf(fp3,  "%d",m_UpdateTime.secsOfWeek);
		//	m_GLSInfo.StateP.write(fp3);
		//	fprintf(fp3, "\n");
		//}
		//static FILE* fp4 = fopen("Phi.txt", "w");
		//if (fp4 && (m_UpdateTime.fracOfSec <= 0.00)) {
		//	static int header_written = 0;
		//	if (!header_written) {
		//		fprintf(fp4, "WeekSecond Phi\n");
		//		header_written = 1;
		//	}
		//	fprintf(fp4, "%d ",m_UpdateTime.secsOfWeek);
		//	m_GLSInfo.SINSPhi.write(fp4);
		//	fprintf(fp4, "\n");
		//}
		//lj修改 2026-08-11：不再按整数秒结束累计区间，改由成功GNSS位置更新触发
		m_GLS.KFPrediction(m_SINSMech.m_DeltTime, m_GLSInfo.SINSCF, m_GLSInfo.SINSQc, m_GLSInfo.StateP, m_GLSInfo.SINSPhi, m_GLSInfo.SINSPPre);
		m_GLSInfo.StateP = m_GLSInfo.SINSPPre;

		// 3. 累乘RTS矩阵
		m_GLSInfo.RTS_StatePhiT.transpose(m_GLSInfo.SINSPhi);
		m_GLSInfo.RTS_PPre = m_GLSInfo.RTS_Ck;
		m_GLSInfo.RTS_Ck.multiply(m_GLSInfo.RTS_PPre, m_GLSInfo.RTS_StatePhiT);
		if (m_bELCData || m_bStateUpdate) // 最后一次更新
		{
			m_GLSInfo.SINSPPre.symm();
			m_GLSInfo.RTS_PPre = m_GLSInfo.SINSPPre;
			m_GLSInfo.RTS_PPre.inv_sp();
			m_GLSInfo.RTS_StatePhiT = m_GLSInfo.RTS_Ck;
			m_GLSInfo.RTS_Ck.multiply(m_GLSInfo.RTS_StatePhiT, m_GLSInfo.RTS_PPre);
			m_GLSInfo.RTS_PPre = m_GLSInfo.SINSPPre;
			m_GLSInfo.RTS_bStart = true;
		}

		// 4. 更新机械编排结果
		M31EQU(m_SINSMech.m_Position, m_GLSInfo.Position);
		M31EQU(m_SINSMech.m_Velocity, m_GLSInfo.Velocity);
		M33EQU(m_SINSMech.m_Rbn, m_GLSInfo.Rbn);
		M31EQU(m_pIMUFFStream->m_bAccel, m_GLSInfo.bAccelCF);
		M31EQU(m_pIMUFFStream->m_bGyros, m_GLSInfo.bGyrosCF);

		return true;
	}


	void CLooseCouplingBase::UpdateClosedLoop()
	{
		if (m_bELCData == false || m_pSINSSol->m_IMUFlag == ELCTYPE_IMU || m_pSINSOpt->m_bClosedLoop == false) return; // 当前无更新,返回

		setSINSKF_ClosedLoop(m_pSINSOpt, &m_GLSInfo, m_pIMUFFStream);
	}


	bool CLooseCouplingBase::PrepareLC()
	{
		m_pSINSSol->m_IMUFlag = 0;

		// 1. 控制IMU中断,IMUFFStream已经内插,但还是要控制,以免中断时间过大导致内插不可靠
		if (m_pIMUFFStream->m_bGap && m_pIMUFFStream->m_GapTime > m_pSINSOpt->m_IMUDataGapTol)
		{
			m_ProcFlag = 2;
			m_ProcInfo = string("IMU Data Gap");
			return false;
		}

		// 2. 确定当前是否进行状态定点更新
		if (m_bELCData) // 如果有观测更新,则状态定点更新一定需要
		{
			m_bStateUpdate = true;
			if (m_bReverse == true && (m_UpdateTime < m_StateTime || fabs(m_UpdateTime - m_StateTime) < 1e-8))
			{
				m_StateTime -= m_UpdateDeltT;
			}
			else if (m_bReverse == false && (m_UpdateTime > m_StateTime || fabs(m_UpdateTime - m_StateTime) < 1e-8))
			{
				m_StateTime += m_UpdateDeltT;
			}
		}
		else
		{
			if (m_bReverse == true && (m_pIMUFFStream->m_time < m_StateTime || fabs(m_pIMUFFStream->m_time - m_StateTime) < 1e-8))
			{
				m_UpdateTime = m_StateTime;
				m_pIMUFFStream->setMeasUpdate(m_StateTime);
				m_bStateUpdate = true;
				m_StateTime -= m_UpdateDeltT;
			}
			else if (m_bReverse == false && (m_pIMUFFStream->m_time > m_StateTime || fabs(m_pIMUFFStream->m_time - m_StateTime) < 1e-8))
			{
				m_UpdateTime = m_StateTime;
				m_pIMUFFStream->setMeasUpdate(m_StateTime);
				m_bStateUpdate = true;
				m_StateTime += m_UpdateDeltT;
			}
			else
			{
				m_bStateUpdate = false;
			}
		}
		
		if (m_bStateUpdate)
		{
			M31Zero(m_pSINSSol->m_MisclosurePos);
		}
		
		// 3. IPS界面时间输出
		// lj毕设：m_UpdateTime
		if (m_bELCData || m_bStateUpdate) DUMP_GETTIME(m_UpdateTime);

		// 4. 取得数据
		if(m_bELCData)
		{
			m_GLSInfo.bMeasPOS = false;
			m_GLSInfo.bMeasVEL = false;
			m_GLSInfo.bMeasATT = false;
			m_GLSInfo.bMeasDMI = false;
			m_GLSInfo.bMeasHED = false;
			m_GLSInfo.bMeasZUP = false;
			m_GLSInfo.bMeasZIH = false;
			m_GLSInfo.bMeasCUP = false;
			m_GLSInfo.bMeasDIS = false;
			m_GLSInfo.bMeasNHC = false;
			m_GLSInfo.bMeasHGH = false;

			m_GLSInfo.MeasDMIPre = m_pELCFFStream->m_ELCData.DMI_DATA.DMIVel;
			if (isEqual(m_GLSInfo.MeasDMIPre, 0.0)) m_GLSInfo.MeasDMIPre = m_pELCFFStream->m_ELCData.DMI_DATA.DMIVel; // ????

			// 判断当前观测更新
			if ((m_pELCFFStream->m_ELCData.ELC_Type&ELCTYPE_POS) && m_pSINSOpt->m_bPOSUsed) m_GLSInfo.bMeasPOS = true;
			if ((m_pELCFFStream->m_ELCData.ELC_Type&ELCTYPE_VEL) && m_pSINSOpt->m_bVELUsed) m_GLSInfo.bMeasVEL = true;
			if ((m_pELCFFStream->m_ELCData.ELC_Type&ELCTYPE_ATT) && m_pSINSOpt->m_bATTUsed) m_GLSInfo.bMeasATT = true;
			if ((m_pELCFFStream->m_ELCData.ELC_Type&ELCTYPE_DMI) && m_pSINSOpt->m_bDMIUsed) m_GLSInfo.bMeasDMI = true;
			if ((m_pELCFFStream->m_ELCData.ELC_Type&ELCTYPE_HED) && m_pSINSOpt->m_bHEDUsed) m_GLSInfo.bMeasHED = true;
			if ((m_pELCFFStream->m_ELCData.ELC_Type&ELCTYPE_ZUP) && m_pSINSOpt->m_bZUPUsed) m_GLSInfo.bMeasZUP = true;
			if ((m_pELCFFStream->m_ELCData.ELC_Type&ELCTYPE_ZIH) && m_pSINSOpt->m_bZIHUsed) m_GLSInfo.bMeasZIH = true;
			if ((m_pELCFFStream->m_ELCData.ELC_Type&ELCTYPE_CUP) && m_pSINSOpt->m_bCUPUsed) m_GLSInfo.bMeasCUP = true;
			if ((m_pELCFFStream->m_ELCData.ELC_Type&ELCTYPE_NHC) && m_pSINSOpt->m_bNHCUsed) m_GLSInfo.bMeasNHC = true;
			if ((m_pELCFFStream->m_ELCData.ELC_Type&ELCTYPE_HGH) && m_pSINSOpt->m_bHGHUsed) m_GLSInfo.bMeasHGH = true;
			if (m_pSINSOpt->m_ZIHRbSyncZUPT) m_GLSInfo.bMeasZIH = m_GLSInfo.bMeasZUP;
			// M33Scale(SQR(1.5), m_pELCFFStream->m_ELCData.GPSPosP);
		}
		
		// 5. KF状态预报
		bool flag = KFPre();
		
		if (m_bELCData == false && m_bStateUpdate == false) m_ProcFlag = 1; // 处于纯惯导状态

		if(m_ProcFlag <= 1) m_pSINSSol->m_IMUFlag = ELCTYPE_IMU;

		return flag;
	}


	void CLooseCouplingBase::UpdateSolution_GLSInfo()
	{
		// 1. 得到位置,速度,旋转矩阵
		M31_M31(m_SINSMech.m_Position, m_GLSInfo.StateX.x + m_GLSInfo.IPos, m_GLSInfo.Position);
		// lj毕设：输出m_GLSInfo.Position
		M31_M31(m_SINSMech.m_Velocity, m_GLSInfo.StateX.x + m_GLSInfo.IVel, m_GLSInfo.Velocity);
		M31EQU(m_GLSInfo.StateX.x + m_GLSInfo.IMisAngle, m_GLSInfo.MisAngle);
		CorrectMisAngle(m_SINSMech.m_Rbn, m_GLSInfo.MisAngle, m_GLSInfo.Rbn);
		UnitOrthoRotMatrix(m_GLSInfo.Rbn);
		M31Zero(m_GLSInfo.MisAngle);  // 之后把失准角清零，方便后面重新得到修正后的失准角

		// 2. 得到姿态角
		double LLH[3] = { 0.0 };
		XYZ2LLH(m_GLSInfo.Position, LLH);

		if (m_pSINSOpt->m_NaviSys == NAVI_E)
		{
			Rbe2Attitude(m_GLSInfo.Rbn, LLH, m_GLSInfo.Attitude);
		}
		else if (m_pSINSOpt->m_NaviSys == NAVI_L)
		{
			Rbl2Attitude(m_GLSInfo.Rbn, m_GLSInfo.Attitude);
		}
		else
		{
			MatrixZero(3, 1, m_GLSInfo.Attitude);
		}

		// 3. 得到零偏
		if (m_pSINSOpt->m_bBiasFedBack)
		{
			M31M31(m_GLSInfo.StateX.x + m_GLSInfo.IAccelBias, m_GLSInfo.AccelBias);
			M31M31(m_GLSInfo.StateX.x + m_GLSInfo.IGyrosBias, m_GLSInfo.GyrosBias);

			// 比例因子,开机误差
			if (m_pSINSOpt->m_bAddAccelBiasScale) M31M31(m_GLSInfo.StateX.x + m_GLSInfo.IAccelBiasScale, m_GLSInfo.AccelBiasScale);
			if (m_pSINSOpt->m_bAddGyrosBiasScale) M31M31(m_GLSInfo.StateX.x + m_GLSInfo.IGyrosBiasScale, m_GLSInfo.GyrosBiasScale);
			if (m_pSINSOpt->m_bAddAccelBiasTob)   M31M31(m_GLSInfo.StateX.x + m_GLSInfo.IAccelBiasTob, m_GLSInfo.AccelBiasTob);
			if (m_pSINSOpt->m_bAddGyrosBiasTob)   M31M31(m_GLSInfo.StateX.x + m_GLSInfo.IGyrosBiasTob, m_GLSInfo.GyrosBiasTob);

			for (int i = 0; i < 6; i++)
			{
				if (m_pSINSOpt->m_bAddAccelBiasOrthog) m_GLSInfo.AccelBiasOrthog[i] += m_GLSInfo.StateX.x[m_GLSInfo.IAccelBiasOrthog + i];
				if (m_pSINSOpt->m_bAddGyrosBiasOrthog) m_GLSInfo.GyrosBiasOrthog[i] += m_GLSInfo.StateX.x[m_GLSInfo.IGyrosBiasOrthog + i];
			}
		}
		else
		{
			M31EQU(m_GLSInfo.StateX.x + m_GLSInfo.IAccelBias, m_GLSInfo.AccelBias);
			M31EQU(m_GLSInfo.StateX.x + m_GLSInfo.IGyrosBias, m_GLSInfo.GyrosBias);

			// 比例因子,开机误差
			if (m_pSINSOpt->m_bAddAccelBiasScale) M31EQU(m_GLSInfo.StateX.x + m_GLSInfo.IAccelBiasScale, m_GLSInfo.AccelBiasScale);
			if (m_pSINSOpt->m_bAddGyrosBiasScale) M31EQU(m_GLSInfo.StateX.x + m_GLSInfo.IGyrosBiasScale, m_GLSInfo.GyrosBiasScale);
			if (m_pSINSOpt->m_bAddAccelBiasTob)   M31EQU(m_GLSInfo.StateX.x + m_GLSInfo.IAccelBiasTob, m_GLSInfo.AccelBiasTob);
			if (m_pSINSOpt->m_bAddGyrosBiasTob)   M31EQU(m_GLSInfo.StateX.x + m_GLSInfo.IGyrosBiasTob, m_GLSInfo.GyrosBiasTob);

			for (int i = 0; i < 6; i++)
			{
				if (m_pSINSOpt->m_bAddAccelBiasOrthog) m_GLSInfo.AccelBiasOrthog[i] = m_GLSInfo.StateX.x[m_GLSInfo.IAccelBiasOrthog + i];
				if (m_pSINSOpt->m_bAddGyrosBiasOrthog) m_GLSInfo.GyrosBiasOrthog[i] = m_GLSInfo.StateX.x[m_GLSInfo.IGyrosBiasOrthog + i];
			}
		}
	}


	void CLooseCouplingBase::UpdateSolution()
	{
		if (m_bStateUpdate == false) return;


		// 1. 更新GLSInfo
		UpdateSolution_GLSInfo();

		static FILE* fpState = fopen("StateX_new.txt", "w");
		if (fpState) {
			static int header_written = 0;
			if (!header_written) {
				fprintf(fpState, "Time(s) StateX\n");
				header_written = 1;
			}

			int time = m_UpdateTime.secsOfWeek + 1;

			double state[15] = {
				m_GLSInfo.Position[0], m_GLSInfo.Position[1], m_GLSInfo.Position[2],
				m_GLSInfo.Velocity[0], m_GLSInfo.Velocity[1], m_GLSInfo.Velocity[2],
				m_GLSInfo.Attitude[0], m_GLSInfo.Attitude[1], m_GLSInfo.Attitude[2],
				m_GLSInfo.AccelBias[0], m_GLSInfo.AccelBias[1], m_GLSInfo.AccelBias[2],
				m_GLSInfo.GyrosBias[0], m_GLSInfo.GyrosBias[1], m_GLSInfo.GyrosBias[2]
			};

			// ===== 第一行 =====
			// Position[0] → 普通小数
			fprintf(fpState, "%d %.7f\n", time, state[0]);

			// ===== 后续数据 =====
			for (int i = 1; i < 15; i++) {

				if (i < 3) {
					// Position[1], Position[2] → 普通小数
					fprintf(fpState, "%6s %.7f\n", "", state[i]);
				}
				else {
					// 其它 → 科学计数法
					fprintf(fpState, "%6s %.7e\n", "", state[i]);
				}
			}

			fflush(fpState);
		}

		// 2. 位置,速度,姿态赋值
		m_pSINSSol->m_time = m_UpdateTime;
		M31EQU(m_GLSInfo.Position, m_pSINSSol->m_XYZPos);
		M31EQU(m_GLSInfo.Velocity, m_pSINSSol->m_XYZVel);
		M31EQU(m_GLSInfo.MisAngle, m_pSINSSol->m_MisAngle);
		M33EQU(m_GLSInfo.Rbn, m_pSINSSol->m_Rbn);
		XYZ2LLH(m_pSINSSol->m_XYZPos, m_pSINSSol->m_LLHPos);
		

		// 3. 得到姿态角
		if (m_pSINSOpt->m_bAzimuth)
		{
			Attitude2Azimuth(m_GLSInfo.Attitude, m_pSINSSol->m_Att);
		}
		else
		{
			M31EQU(m_GLSInfo.Attitude, m_pSINSSol->m_Att);
		}


		// 4. 得到零偏
		for (int i = 0; i < 6; i++)
		{
			if (i < 3)
			{
				m_pSINSSol->m_AccelBias[i] = m_GLSInfo.AccelBias[i];
				m_pSINSSol->m_AccelBiasScale[i] = m_GLSInfo.AccelBiasScale[i];
				m_pSINSSol->m_AccelBiasTob[i] = m_GLSInfo.AccelBiasTob[i];

				m_pSINSSol->m_GyrosBias[i] = m_GLSInfo.GyrosBias[i];
				m_pSINSSol->m_GyrosBiasScale[i] = m_GLSInfo.GyrosBiasScale[i];
				m_pSINSSol->m_GyrosBiasTob[i] = m_GLSInfo.GyrosBiasTob[i];
			}

			m_pSINSSol->m_AccelBiasOrthog[i] = m_GLSInfo.AccelBiasOrthog[i];
			m_pSINSSol->m_GyrosBiasOrthog[i] = m_GLSInfo.GyrosBiasOrthog[i];
		}

		// 5. 得到状态协方差
		MatrixCopySub(m_GLSInfo.IPos     , m_GLSInfo.IPos     , 3, 3, m_GLSInfo.StateP.row, m_GLSInfo.StateP.col, m_GLSInfo.StateP.x, m_pSINSSol->m_XYZPosP);
		MatrixCopySub(m_GLSInfo.IVel     , m_GLSInfo.IVel     , 3, 3, m_GLSInfo.StateP.row, m_GLSInfo.StateP.col, m_GLSInfo.StateP.x, m_pSINSSol->m_XYZVelP);
		MatrixCopySub(m_GLSInfo.IMisAngle, m_GLSInfo.IMisAngle, 3, 3, m_GLSInfo.StateP.row, m_GLSInfo.StateP.col, m_GLSInfo.StateP.x, m_pSINSSol->m_MisAngleP);

		if (m_pSINSOpt->m_NaviSys == NAVI_E)
		{
			getLocalAttitudeP(m_pSINSSol->m_LLHPos, m_GLSInfo.Rbn, m_pSINSSol->m_MisAngleP, m_pSINSSol->m_AttP);
		}
		else if (m_pSINSOpt->m_NaviSys == NAVI_L)
		{
			M33EQU(m_pSINSSol->m_MisAngleP, m_pSINSSol->m_AttP);
		}

		MatrixCopySub(m_GLSInfo.IAccelBias, m_GLSInfo.IAccelBias, 3, 3, m_GLSInfo.StateP.row, m_GLSInfo.StateP.col, m_GLSInfo.StateP.x, m_pSINSSol->m_AccelBiasP);
		MatrixCopySub(m_GLSInfo.IGyrosBias, m_GLSInfo.IGyrosBias, 3, 3, m_GLSInfo.StateP.row, m_GLSInfo.StateP.col, m_GLSInfo.StateP.x, m_pSINSSol->m_GyrosBiasP);
		if (m_pSINSOpt->m_bAddAccelBiasScale) MatrixCopySub(m_GLSInfo.IAccelBiasScale, m_GLSInfo.IAccelBiasScale, 3, 3, m_GLSInfo.StateP.row, m_GLSInfo.StateP.col, m_GLSInfo.StateP.x, m_pSINSSol->m_AccelBiasScaleP);
		if (m_pSINSOpt->m_bAddGyrosBiasScale) MatrixCopySub(m_GLSInfo.IGyrosBiasScale, m_GLSInfo.IGyrosBiasScale, 3, 3, m_GLSInfo.StateP.row, m_GLSInfo.StateP.col, m_GLSInfo.StateP.x, m_pSINSSol->m_GyrosBiasScaleP);
		if (m_pSINSOpt->m_bAddAccelBiasTob)   MatrixCopySub(m_GLSInfo.IAccelBiasTob, m_GLSInfo.IAccelBiasTob, 3, 3, m_GLSInfo.StateP.row, m_GLSInfo.StateP.col, m_GLSInfo.StateP.x, m_pSINSSol->m_AccelBiasTobP);
		if (m_pSINSOpt->m_bAddGyrosBiasTob)   MatrixCopySub(m_GLSInfo.IGyrosBiasTob, m_GLSInfo.IGyrosBiasTob, 3, 3, m_GLSInfo.StateP.row, m_GLSInfo.StateP.col, m_GLSInfo.StateP.x, m_pSINSSol->m_GyrosBiasTobP);

		if (m_pSINSOpt->m_bAddAccelBiasOrthog)
		{
			MatrixCopySub(m_GLSInfo.IAccelBiasOrthog, m_GLSInfo.IAccelBiasOrthog, 6, 6, m_GLSInfo.StateP.row, m_GLSInfo.StateP.col, m_GLSInfo.StateP.x, m_pSINSSol->m_AccelBiasOrthogP);
		}
		if (m_pSINSOpt->m_bAddGyrosBiasOrthog)
		{
			MatrixCopySub(m_GLSInfo.IGyrosBiasOrthog, m_GLSInfo.IGyrosBiasOrthog, 6, 6, m_GLSInfo.StateP.row, m_GLSInfo.StateP.col, m_GLSInfo.StateP.x, m_pSINSSol->m_GyrosBiasOrthogP);
		}


		// 6. 得到其它信息
		if (m_pSINSSol->m_IMUFlag&ELCTYPE_POS)
		{
			m_pSINSSol->m_Quality = getQualityNumber(m_pSINSSol->m_XYZPosP);
			if (m_pELCFFStream->m_ELCData.GNSS_DATA.Quality > 0 && m_pELCFFStream->m_ELCData.GNSS_DATA.Quality < 7) m_pSINSSol->m_Quality = m_pELCFFStream->m_ELCData.GNSS_DATA.Quality;
		} 
		else if (fabs(m_pSINSSol->m_time - m_PosUpt_Pre) > 3)
		{
			m_pSINSSol->m_Quality = getQualityNumber(m_pSINSSol->m_XYZPosP);
		}
		if (m_pSINSSol->m_Quality == 0) m_pSINSSol->m_Quality = 6;
		m_pSINSSol->m_bKS = (m_pSINSSol->m_IMUFlag&ELCTYPE_ZUP) ? 2 : 1;

		m_pSINSSol->MFSINSAssign(&m_GLSInfo);

		// 如果是松组合,则输出POS中的模糊度固定信息
		if (m_bSTC == false && m_pELCFFStream->m_ELCData.ELC_Type&ELCTYPE_POS)
		{
			m_pSINSSol->m_ZTD = m_pELCFFStream->m_ELCData.GNSS_DATA.ZTD;
			m_pSINSSol->m_ZWD = m_pELCFFStream->m_ELCData.GNSS_DATA.ZWD;
			m_pSINSSol->m_ZWDP = m_pELCFFStream->m_ELCData.GNSS_DATA.ZWDP;
			m_pSINSSol->m_nSat = m_pELCFFStream->m_ELCData.GNSS_DATA.nSat;
			m_pSINSSol->m_nSat_ARHold = m_pELCFFStream->m_ELCData.GNSS_DATA.nSat_ARHold;

			for (int i = 0; i < NSYS; i++)
			{
				m_pSINSSol->m_GNSSClk[i] = m_pELCFFStream->m_ELCData.GNSS_DATA.GNSSClk[i];
				m_pSINSSol->m_GNSSClkP[i] = m_pELCFFStream->m_ELCData.GNSS_DATA.GNSSClkP[i];
				m_pSINSSol->m_fgnss[i] = m_pELCFFStream->m_ELCData.GNSS_DATA.fgnss[i];
				m_pSINSSol->m_ngnss[i] = m_pELCFFStream->m_ELCData.GNSS_DATA.ngnss[i];
				m_pSINSSol->m_vgnss[i] = m_pELCFFStream->m_ELCData.GNSS_DATA.vgnss[i];
			}

			m_pSINSSol->m_SolType = m_pELCFFStream->m_ELCData.GNSS_DATA.SolType;
			m_pSINSSol->m_AmbState = m_pELCFFStream->m_ELCData.GNSS_DATA.AmbState;
			m_pSINSSol->m_DDOP = m_pELCFFStream->m_ELCData.GNSS_DATA.DDOP;
			m_pSINSSol->m_PDOP = m_pELCFFStream->m_ELCData.GNSS_DATA.PDOP;
			m_pSINSSol->m_HDOP = m_pELCFFStream->m_ELCData.GNSS_DATA.HDOP;
			m_pSINSSol->m_VDOP = m_pELCFFStream->m_ELCData.GNSS_DATA.VDOP;
			m_pSINSSol->m_ARRatio = m_pELCFFStream->m_ELCData.GNSS_DATA.ARRatio;
			m_pSINSSol->m_RMSL = m_pELCFFStream->m_ELCData.GNSS_DATA.RMSL;
			m_pSINSSol->m_RMSP = m_pELCFFStream->m_ELCData.GNSS_DATA.RMSP;
			m_pSINSSol->m_SigL = m_pELCFFStream->m_ELCData.GNSS_DATA.SigL;
			m_pSINSSol->m_SigP = m_pELCFFStream->m_ELCData.GNSS_DATA.SigP;
		}

		if (MatrixNorm2(3, 1, m_pSINSOpt->m_GNSSOpt.m_BaseXYZ[m_pELCFFStream->m_ELCData.GNSS_DATA.BaseID]) > 0.001)
		{
			double dXYZ[3] = { 0.0 }, BaseLLH[3] = { 0.0 };
			m_pSINSSol->m_BaseID = m_pELCFFStream->m_ELCData.GNSS_DATA.BaseID;
			M31_M31(m_pSINSSol->m_XYZPos, m_pSINSOpt->m_GNSSOpt.m_BaseXYZ[m_pSINSSol->m_BaseID], dXYZ);
			XYZ2LLH(m_pSINSOpt->m_GNSSOpt.m_BaseXYZ[m_pSINSSol->m_BaseID], BaseLLH);
			Vxyz2enu(BaseLLH, dXYZ, m_pSINSSol->m_BLENU);
		}

		if (m_bELCData || fmod(_TOW(m_UpdateTime), m_MeasDeltT) == 0.0)
		{
			m_pSINSSol->writeSolution();  // 输出文本结果
		}
		else
		{
			m_pSINSSol->writeSolution(true);  // 输出二进制结果
		}
	}


	bool CLooseCouplingBase::runOneSubLC()
	{
		if (m_bELCData == false) return true;

		// 1. 观测值更新优先级
		int nOrder = 4;
		int MeasUPTOrder[10];
		int ELCType = 0;
		string str_type[4] = { "CUP", "ZUP", "POS", "DMI|NHC" };
		MeasUPTOrder[0] = ELCTYPE_CUP;
		MeasUPTOrder[1] = ELCTYPE_ZUP;
		MeasUPTOrder[2] = ELCTYPE_POS;
		MeasUPTOrder[3] = ELCTYPE_DMI | ELCTYPE_NHC;

		bool bPassUPT = false;

		for (int i = 0; i < nOrder; i++)
		{
			bPassUPT = false;

			ELCType = MeasUPTOrder[i];

			// 得到观测更新对应的sigma阈值
			double sig_thres = getQCSigThres(ELCType, m_pSINSOpt);
			double sig = 0.0;

			// 松组合迭代次数越多,有时候会越靠近观测值,导致残差减小
			// 无法进行更新拒绝,一般2-3次就可以。没有失锁的话,迭代1次就好
			// 失锁长了,迭代1-2次,减小线性话误差,第3次用来检验
			int nit = 7;
			int it = 0;
			double dl = 0.0;
			double xyz0[3] = { 0.0 };
			double xyz1[3] = { 0.0 };
			double dxyz[3] = { 0.0 };

			// 使用不同的观测值更新,其状态都一样,在KF将不做预报,这里的X和XP就是预报值
			m_GLSInfo.StateX.zero();

			// 起始和末尾不做零速更新
			if ((m_pELCFFStream->m_time < m_pSINSOpt->m_MoveBeginTime || m_pELCFFStream->m_time > m_pSINSOpt->m_MoveEndTime) && (ELCType&ELCTYPE_ZUP))
			{
				continue;
			}

			// 验后剔除错误位置应该放在这里
			if (m_pELCFFStream->m_ELCData.GNSS_DATA.Quality > 6 && (ELCType&ELCTYPE_POS))
			{
				continue;
			}

			// 2. 每种观测值进行迭代滤波更新
			while (it++ < nit)
			{
				if (it <= 2)      dl = 0.01;
				else if (it <= 4) dl = 0.10;
				else if (it <= 6) dl = 0.40;
				else              dl = 0.50;

				// 得到观测方程
				if (setMEQ(it, ELCType) == false) break;

				m_GLS.m_StatePp = m_GLSInfo.SINSPPre; // 每次迭代前,状态方差都用最原始的
				m_GLS.m_StateXp = m_GLSInfo.StateX;

				// lj毕设输出
				//static FILE* fp5 = fopen("H.txt", "w");
				//if (fp5) {
				//	static int header_written = 0;
				//	if (!header_written) {
				//		fprintf(fp5, "WeekSecond H\n");
				//		header_written = 1;
				//	}
				//	fprintf(fp5,"%d",m_UpdateTime.secsOfWeek);
				//	m_GLS.m_H.write(fp5);
				//	fprintf(fp5, "\n");
				//}
				//static FILE* fp6 = fopen("R.txt", "w");
				//if (fp6) {
				//	static int header_written = 0;
				//	if (!header_written) {
				//		fprintf(fp6, "WeekSecond H\n");
				//		header_written = 1;
				//	}
				//	fprintf(fp6,  "%d",m_UpdateTime.secsOfWeek);
				//	m_GLS.m_R.write(fp6);
				//	fprintf(fp6, "\n");
				//}
				//static FILE* fp7 = fopen("Inno.txt", "w");
				//if (fp7) {
				//	static int header_written = 0;
				//	if (!header_written) {
				//		fprintf(fp7, "WeekSecond H\n");
				//		header_written = 1;
				//	}
				//	fprintf(fp7,  "%d",m_UpdateTime.secsOfWeek);
				//	m_GLS.m_Inno.write(fp7);
				//	fprintf(fp7, "\n");
				//}

				if (m_GLS.GLS() == false) break;
				m_GLSInfo.StateX = m_GLS.m_StateX;
				m_GLSInfo.StateP = m_GLS.m_StateP;

				//static FILE* fp2 = fopen("stateX.txt", "w");
				//if (fp2 && (m_UpdateTime.fracOfSec <= 0.00)) {
				//	static int header_written = 0;
				//	if (!header_written) {
				//		fprintf(fp2, "Time StateX\n");
				//		header_written = 1;
				//	}
				//	int time = m_UpdateTime.secsOfWeek + 1;
				//	fprintf(fp2, "%d", time);
				//	m_GLSInfo.StateX.write(fp2);
				//	fprintf(fp2, "\n");
				//}

				// sigma检验通过时迭代结束
				if ((sig = m_GLS.m_ResiSigma0) <= sig_thres) { bPassUPT = true; break; }

				// 改正数小于一定范围时迭代结束,并且判断是否需要强制更新
				M31EQU(m_GLS.m_StateX.x, xyz1);
				M31_M31(xyz1, xyz0, dxyz);
				M31EQU(xyz1, xyz0);

				if (MatrixNorm2(3, 1, dxyz) < dl)// 阈值设为0.01m好些
				{
					// 观测值预报方差大于观测噪声时，强制观测更新，但如果是位置直接强制更新
					if (!(ELCType&ELCTYPE_POS) && m_GLS.m_bMeasVarPre) { bPassUPT = true; break; }
					else if (ELCType&ELCTYPE_POS) { bPassUPT = true; break; }
					break;
				}
			}

			// 更新成功,直接退出
			if (bPassUPT) break;

			if (m_GLSInfo.nMeas_ELC > 0 && sig > 0.0)
			{
				char msg[MAXSIZE] = { '\0' };
				sprintf(msg, "Reject %s Updating with sigma: %.2f.", str_type[i].c_str(), sig);
				DumpException(string(msg), LOGT_INFO, DUMP_LOCATION);
			}
		}

		if (ELCType&ELCTYPE_POS) m_PosUpt_Pre = m_pELCFFStream->m_ELCData.gt;

		// 更新失败,仍采用预报值
		if (bPassUPT == false)
		{
			m_GLSInfo.StateX.zero();
			m_GLSInfo.StateP = m_GLSInfo.SINSPPre;
			ELCType = ELCTYPE_IMU;
		}
		else
		{
			setMEQ(0, ELCType); // 得到位置的验后残差
		}

		//lj修改 2026-08-11：只有通过质控的GNSS位置更新才结束并输出累计区间
		if (bPassUPT && (ELCType&ELCTYPE_POS))
		{
			m_GLS.updateTime = m_UpdateTime.secsOfWeek;
			m_GLS.FinishGNSSInterval(m_GLSInfo.SINSPPre);
		}

		m_pSINSSol->m_IMUFlag = ELCType;

		return true;
	}


	int CLooseCouplingBase::runOneLC(GPSTIME UpdateTime, bool bELCData)
	{
		if (UpdateTime == m_UpdateTime) return 0;

		if (_TOW(m_UpdateTime) == 449815)
		{
			printf("");
		}

		m_ProcFlag = 0;
		m_UpdateTime = UpdateTime;
		m_bELCData = bELCData;

		if (PrepareLC() == false)
		{
			MatrixZero(3, 1, m_GLSInfo.bAccelCF);
			MatrixZero(3, 1, m_GLSInfo.bGyrosCF);
			return m_ProcFlag;
		}

		if (m_ProcFlag > 0) return m_ProcFlag;

		runOneSubLC();

		// 以下操作顺序不能改变

		// lj毕设：在此处函数中输出位置
		// 1. 必须在UpdateClosedLoop之前执行
		UpdateSolution();

		// 2. GPS outage 时是否采用闭环
		UpdateClosedLoop();

		// 3. 置为false,以免对下一时刻有影响
		if (m_bStateUpdate) m_bStateUpdate = false;

		return m_ProcFlag;
	}


	int CLooseCouplingBase::runOneTCSemi(GPSTIME UpdateTime, bool bELCData)
	{
		m_ProcFlag = 0;
		m_UpdateTime = UpdateTime;
		m_bELCData = bELCData;

		if (m_bStateUpdate_STC)
		{
			if (PrepareLC() == false)
			{
				MatrixZero(3, 1, m_GLSInfo.bAccelCF);
				MatrixZero(3, 1, m_GLSInfo.bGyrosCF);
				return m_ProcFlag;
			}

			if (m_ProcFlag > 0) return m_ProcFlag;
		}

		if (m_bELCData_STC)
		{
			runOneSubLC();

			// 以下操作顺序不能改变

			// 1. 必须在UpdateClosedLoop之前执行
			UpdateSolution();

			// 2. GPS outage 时是否采用闭环
			UpdateClosedLoop();

			// 3. 置为false,以免对下一时刻有影响
			if (m_bStateUpdate) m_bStateUpdate = false;
		}

		return m_ProcFlag;
	}


	/**
	* @brief       设置SINS-KF的初始状态
	* @param[in]   pSINSOpt     CSINSOption
	* @param[out]  pGLSInfo     SINS_GLSINFO
	* @return      void
	* @note        松组合/紧组合共用
	* @par History:
	*              2018/01/17,Feng Zhu, new \n
	* @internals
	*/
	void CLooseCouplingBase::setSINSKF_InitialState(const CSINSOption* pSINSOpt, SINS_GLSINFO* pGLSInfo)
	{
		// 初始化赋零,对第二次初始化有用
		pGLSInfo->StateP.zero();

		pGLSInfo->StateP.at(pGLSInfo->IPos + 0, pGLSInfo->IPos + 0, SQR(pSINSOpt->m_SSP.ISDV_DeltPosX));
		pGLSInfo->StateP.at(pGLSInfo->IPos + 1, pGLSInfo->IPos + 1, SQR(pSINSOpt->m_SSP.ISDV_DeltPosY));
		pGLSInfo->StateP.at(pGLSInfo->IPos + 2, pGLSInfo->IPos + 2, SQR(pSINSOpt->m_SSP.ISDV_DeltPosZ));

		pGLSInfo->StateP.at(pGLSInfo->IVel + 0, pGLSInfo->IVel + 0, SQR(pSINSOpt->m_SSP.ISDV_DeltVelX));
		pGLSInfo->StateP.at(pGLSInfo->IVel + 1, pGLSInfo->IVel + 1, SQR(pSINSOpt->m_SSP.ISDV_DeltVelY));
		pGLSInfo->StateP.at(pGLSInfo->IVel + 2, pGLSInfo->IVel + 2, SQR(pSINSOpt->m_SSP.ISDV_DeltVelZ));

		pGLSInfo->StateP.at(pGLSInfo->IMisAngle + 0, pGLSInfo->IMisAngle + 0, SQR(pSINSOpt->m_SSP.ISDV_MisAngleX));
		pGLSInfo->StateP.at(pGLSInfo->IMisAngle + 1, pGLSInfo->IMisAngle + 1, SQR(pSINSOpt->m_SSP.ISDV_MisAngleY));
		pGLSInfo->StateP.at(pGLSInfo->IMisAngle + 2, pGLSInfo->IMisAngle + 2, SQR(pSINSOpt->m_SSP.ISDV_MisAngleZ));

		pGLSInfo->StateP.at(pGLSInfo->IAccelBias + 0, pGLSInfo->IAccelBias + 0, SQR(pSINSOpt->m_SSP.ISDV_AccelBiasX));
		pGLSInfo->StateP.at(pGLSInfo->IAccelBias + 1, pGLSInfo->IAccelBias + 1, SQR(pSINSOpt->m_SSP.ISDV_AccelBiasY));
		pGLSInfo->StateP.at(pGLSInfo->IAccelBias + 2, pGLSInfo->IAccelBias + 2, SQR(pSINSOpt->m_SSP.ISDV_AccelBiasZ));

		pGLSInfo->StateP.at(pGLSInfo->IGyrosBias + 0, pGLSInfo->IGyrosBias + 0, SQR(pSINSOpt->m_SSP.ISDV_GyrosBiasX));
		pGLSInfo->StateP.at(pGLSInfo->IGyrosBias + 1, pGLSInfo->IGyrosBias + 1, SQR(pSINSOpt->m_SSP.ISDV_GyrosBiasY));
		pGLSInfo->StateP.at(pGLSInfo->IGyrosBias + 2, pGLSInfo->IGyrosBias + 2, SQR(pSINSOpt->m_SSP.ISDV_GyrosBiasZ));


		// 过程噪声
		pGLSInfo->SINSQc.at(pGLSInfo->IPos + 0, pGLSInfo->IPos + 0, SQR(pSINSOpt->m_SSP.PNSD_PosX));
		pGLSInfo->SINSQc.at(pGLSInfo->IPos + 1, pGLSInfo->IPos + 1, SQR(pSINSOpt->m_SSP.PNSD_PosY));
		pGLSInfo->SINSQc.at(pGLSInfo->IPos + 2, pGLSInfo->IPos + 2, SQR(pSINSOpt->m_SSP.PNSD_PosZ));

		pGLSInfo->SINSQc.at(pGLSInfo->IVel + 0, pGLSInfo->IVel + 0, SQR(pSINSOpt->m_SSP.PNSD_VelX));
		pGLSInfo->SINSQc.at(pGLSInfo->IVel + 1, pGLSInfo->IVel + 1, SQR(pSINSOpt->m_SSP.PNSD_VelY));
		pGLSInfo->SINSQc.at(pGLSInfo->IVel + 2, pGLSInfo->IVel + 2, SQR(pSINSOpt->m_SSP.PNSD_VelZ));

		pGLSInfo->SINSQc.at(pGLSInfo->IMisAngle + 0, pGLSInfo->IMisAngle + 0, SQR(pSINSOpt->m_SSP.PNSD_AttX));
		pGLSInfo->SINSQc.at(pGLSInfo->IMisAngle + 1, pGLSInfo->IMisAngle + 1, SQR(pSINSOpt->m_SSP.PNSD_AttY));
		pGLSInfo->SINSQc.at(pGLSInfo->IMisAngle + 2, pGLSInfo->IMisAngle + 2, SQR(pSINSOpt->m_SSP.PNSD_AttZ));

		pGLSInfo->SINSQc.at(pGLSInfo->IAccelBias + 0, pGLSInfo->IAccelBias + 0, SQR(pSINSOpt->m_SSP.PNSD_AccelBiasX));
		pGLSInfo->SINSQc.at(pGLSInfo->IAccelBias + 1, pGLSInfo->IAccelBias + 1, SQR(pSINSOpt->m_SSP.PNSD_AccelBiasY));
		pGLSInfo->SINSQc.at(pGLSInfo->IAccelBias + 2, pGLSInfo->IAccelBias + 2, SQR(pSINSOpt->m_SSP.PNSD_AccelBiasZ));

		pGLSInfo->SINSQc.at(pGLSInfo->IGyrosBias + 0, pGLSInfo->IGyrosBias + 0, SQR(pSINSOpt->m_SSP.PNSD_GyrosBiasX));
		pGLSInfo->SINSQc.at(pGLSInfo->IGyrosBias + 1, pGLSInfo->IGyrosBias + 1, SQR(pSINSOpt->m_SSP.PNSD_GyrosBiasY));
		pGLSInfo->SINSQc.at(pGLSInfo->IGyrosBias + 2, pGLSInfo->IGyrosBias + 2, SQR(pSINSOpt->m_SSP.PNSD_GyrosBiasZ));

		MatrixZero(3, 3, pGLSInfo->DeltPosQc); MatrixZero(3, 3, pGLSInfo->DeltVelQc); MatrixZero(3, 3, pGLSInfo->MisAngleQc);
		pGLSInfo->DeltPosQc[0] = SQR(pSINSOpt->m_SSP.PNSD_PosX);
		pGLSInfo->DeltPosQc[4] = SQR(pSINSOpt->m_SSP.PNSD_PosY);
		pGLSInfo->DeltPosQc[8] = SQR(pSINSOpt->m_SSP.PNSD_PosZ);
		pGLSInfo->DeltVelQc[0] = SQR(pSINSOpt->m_SSP.PNSD_VelX);
		pGLSInfo->DeltVelQc[4] = SQR(pSINSOpt->m_SSP.PNSD_VelY);
		pGLSInfo->DeltVelQc[8] = SQR(pSINSOpt->m_SSP.PNSD_VelZ);
		pGLSInfo->MisAngleQc[0] = SQR(pSINSOpt->m_SSP.PNSD_AttX);
		pGLSInfo->MisAngleQc[4] = SQR(pSINSOpt->m_SSP.PNSD_AttY);
		pGLSInfo->MisAngleQc[8] = SQR(pSINSOpt->m_SSP.PNSD_AttZ);

		if (pSINSOpt->m_bAddAccelBiasScale)
		{
			pGLSInfo->StateP.at(pGLSInfo->IAccelBiasScale + 0, pGLSInfo->IAccelBiasScale + 0, SQR(pSINSOpt->m_SSP.ISDV_AccelBiasScaleX));
			pGLSInfo->StateP.at(pGLSInfo->IAccelBiasScale + 1, pGLSInfo->IAccelBiasScale + 1, SQR(pSINSOpt->m_SSP.ISDV_AccelBiasScaleY));
			pGLSInfo->StateP.at(pGLSInfo->IAccelBiasScale + 2, pGLSInfo->IAccelBiasScale + 2, SQR(pSINSOpt->m_SSP.ISDV_AccelBiasScaleZ));

			pGLSInfo->SINSQc.at(pGLSInfo->IAccelBiasScale + 0, pGLSInfo->IAccelBiasScale + 0, SQR(pSINSOpt->m_SSP.PNSD_AccelBiasScaleX));
			pGLSInfo->SINSQc.at(pGLSInfo->IAccelBiasScale + 1, pGLSInfo->IAccelBiasScale + 1, SQR(pSINSOpt->m_SSP.PNSD_AccelBiasScaleY));
			pGLSInfo->SINSQc.at(pGLSInfo->IAccelBiasScale + 2, pGLSInfo->IAccelBiasScale + 2, SQR(pSINSOpt->m_SSP.PNSD_AccelBiasScaleZ));
		}

		if (pSINSOpt->m_bAddGyrosBiasScale)
		{
			pGLSInfo->StateP.at(pGLSInfo->IGyrosBiasScale + 0, pGLSInfo->IGyrosBiasScale + 0, SQR(pSINSOpt->m_SSP.ISDV_GyrosBiasScaleX));
			pGLSInfo->StateP.at(pGLSInfo->IGyrosBiasScale + 1, pGLSInfo->IGyrosBiasScale + 1, SQR(pSINSOpt->m_SSP.ISDV_GyrosBiasScaleY));
			pGLSInfo->StateP.at(pGLSInfo->IGyrosBiasScale + 2, pGLSInfo->IGyrosBiasScale + 2, SQR(pSINSOpt->m_SSP.ISDV_GyrosBiasScaleZ));

			pGLSInfo->SINSQc.at(pGLSInfo->IGyrosBiasScale + 0, pGLSInfo->IGyrosBiasScale + 0, SQR(pSINSOpt->m_SSP.PNSD_GyrosBiasScaleX));
			pGLSInfo->SINSQc.at(pGLSInfo->IGyrosBiasScale + 1, pGLSInfo->IGyrosBiasScale + 1, SQR(pSINSOpt->m_SSP.PNSD_GyrosBiasScaleY));
			pGLSInfo->SINSQc.at(pGLSInfo->IGyrosBiasScale + 2, pGLSInfo->IGyrosBiasScale + 2, SQR(pSINSOpt->m_SSP.PNSD_GyrosBiasScaleZ));
		}

		if (pSINSOpt->m_bAddAccelBiasTob)
		{
			pGLSInfo->StateP.at(pGLSInfo->IAccelBiasTob + 0, pGLSInfo->IAccelBiasTob + 0, SQR(pSINSOpt->m_SSP.ISDV_AccelBiasTobX));
			pGLSInfo->StateP.at(pGLSInfo->IAccelBiasTob + 1, pGLSInfo->IAccelBiasTob + 1, SQR(pSINSOpt->m_SSP.ISDV_AccelBiasTobY));
			pGLSInfo->StateP.at(pGLSInfo->IAccelBiasTob + 2, pGLSInfo->IAccelBiasTob + 2, SQR(pSINSOpt->m_SSP.ISDV_AccelBiasTobZ));
		}

		if (pSINSOpt->m_bAddGyrosBiasTob)
		{
			pGLSInfo->StateP.at(pGLSInfo->IGyrosBiasTob + 0, pGLSInfo->IGyrosBiasTob + 0, SQR(pSINSOpt->m_SSP.ISDV_GyrosBiasTobX));
			pGLSInfo->StateP.at(pGLSInfo->IGyrosBiasTob + 1, pGLSInfo->IGyrosBiasTob + 1, SQR(pSINSOpt->m_SSP.ISDV_GyrosBiasTobY));
			pGLSInfo->StateP.at(pGLSInfo->IGyrosBiasTob + 2, pGLSInfo->IGyrosBiasTob + 2, SQR(pSINSOpt->m_SSP.ISDV_GyrosBiasTobZ));
		}

		if (pSINSOpt->m_bAddAccelBiasOrthog)
		{
			for (int i = 0; i < 6; i++)
			{
				pGLSInfo->StateP.at(pGLSInfo->IAccelBiasOrthog + i, pGLSInfo->IAccelBiasOrthog + i, SQR(pSINSOpt->m_SSP.ISDV_AccelBiasOrthog[i]));
				pGLSInfo->SINSQc.at(pGLSInfo->IAccelBiasOrthog + i, pGLSInfo->IAccelBiasOrthog + i, SQR(pSINSOpt->m_SSP.PNSD_AccelBiasOrthog[i]));
			}
		}

		if (pSINSOpt->m_bAddGyrosBiasOrthog)
		{
			for (int i = 0; i < 6; i++)
			{
				pGLSInfo->StateP.at(pGLSInfo->IGyrosBiasOrthog + i, pGLSInfo->IGyrosBiasOrthog, SQR(pSINSOpt->m_SSP.ISDV_GyrosBiasOrthog[i]));
				pGLSInfo->SINSQc.at(pGLSInfo->IGyrosBiasOrthog + i, pGLSInfo->IGyrosBiasOrthog, SQR(pSINSOpt->m_SSP.PNSD_GyrosBiasOrthog[i]));
			}
		}
	}


	/**
	* @brief       设置SINS-KF的过程噪声矩阵,仅常数项
	* @param[in]   pSINSOpt     CSINSOption
	* @param[out]  pGLSInfo     SINS_GLSINFO
	* @return      void
	* @note        松组合/紧组合共用
	* @par History:
	*              2018/01/17,Feng Zhu, new \n
	* @internals
	*/
	void CLooseCouplingBase::setSINSKF_Qc_Const(const CSINSOption* pSINSOpt, SINS_GLSINFO* pGLSInfo)
	{
		///< ToDo(IPS2.2):  在setSINSKF_InitialState中已赋值
	}


	/**
	* @brief       设置SINS-KF的过程噪声矩阵Qc,仅变化项
	* @param[in]   pSINSOpt     CSINSOption
	* @param[out]  pGLSInfo     SINS_GLSINFO
	* @return      void
	* @note        松组合/紧组合共用
	* @par History:
	*              2018/01/17,Feng Zhu, new \n
	* @internals
	*/
	void CLooseCouplingBase::setSINSKF_Qc_Change(const CSINSOption* pSINSOpt, SINS_GLSINFO* pGLSInfo, bool bIMUIn)
	{
		double Qc[9] = { 0.0 };
		double DeltQcTemp[9] = { 0.0 };
		double value = 10000.0;

		// 位置谱密度
		if (bIMUIn)
		{
			// IMU如果是内插数据，放大过程噪声
			for (int i = 0; i < 3; i++) DeltQcTemp[3 * i + i] = pGLSInfo->DeltPosQc[3 * i + i] * value;
			MatrixMultiply_HPHT(3, 3, pGLSInfo->Rbn, DeltQcTemp, true, Qc);
		}
		else
			MatrixMultiply_HPHT(3, 3, pGLSInfo->Rbn, pGLSInfo->DeltPosQc, true, Qc);
		pGLSInfo->SINSQc.setSub(pGLSInfo->IPos, pGLSInfo->IPos, 3, 3, Qc);

		// 速度谱密度
		if (bIMUIn)
		{
			for (int i = 0; i < 3; i++) DeltQcTemp[3 * i + i] = pGLSInfo->DeltVelQc[3 * i + i] * value;
			MatrixMultiply_HPHT(3, 3, pGLSInfo->Rbn, DeltQcTemp, true, Qc);
		}
		else
			MatrixMultiply_HPHT(3, 3, pGLSInfo->Rbn, pGLSInfo->DeltVelQc, true, Qc);
		pGLSInfo->SINSQc.setSub(pGLSInfo->IVel, pGLSInfo->IVel, 3, 3, Qc);

		// 失准角谱密度
		if (bIMUIn)
		{
			for (int i = 0; i < 3; i++) DeltQcTemp[3 * i + i] = pGLSInfo->MisAngleQc[3 * i + i] * value;
			MatrixMultiply_HPHT(3, 3, pGLSInfo->Rbn, DeltQcTemp, true, Qc);
		}
		else
			MatrixMultiply_HPHT(3, 3, pGLSInfo->Rbn, pGLSInfo->MisAngleQc, true, Qc);
		pGLSInfo->SINSQc.setSub(pGLSInfo->IMisAngle, pGLSInfo->IMisAngle, 3, 3, Qc);

		// 零偏
		if (bIMUIn)
		{
			pGLSInfo->SINSQc.at(pGLSInfo->IAccelBias + 0, pGLSInfo->IAccelBias + 0, SQR(pSINSOpt->m_SSP.PNSD_AccelBiasX) * value);
			pGLSInfo->SINSQc.at(pGLSInfo->IAccelBias + 1, pGLSInfo->IAccelBias + 1, SQR(pSINSOpt->m_SSP.PNSD_AccelBiasY) * value);
			pGLSInfo->SINSQc.at(pGLSInfo->IAccelBias + 2, pGLSInfo->IAccelBias + 2, SQR(pSINSOpt->m_SSP.PNSD_AccelBiasZ) * value);

			pGLSInfo->SINSQc.at(pGLSInfo->IGyrosBias + 0, pGLSInfo->IGyrosBias + 0, SQR(pSINSOpt->m_SSP.PNSD_GyrosBiasX) * value);
			pGLSInfo->SINSQc.at(pGLSInfo->IGyrosBias + 1, pGLSInfo->IGyrosBias + 1, SQR(pSINSOpt->m_SSP.PNSD_GyrosBiasY) * value);
			pGLSInfo->SINSQc.at(pGLSInfo->IGyrosBias + 2, pGLSInfo->IGyrosBias + 2, SQR(pSINSOpt->m_SSP.PNSD_GyrosBiasZ) * value);
		}

		if (pSINSOpt->m_NaviSys == NAVI_L)
		{
			pGLSInfo->SINSQc.at(pGLSInfo->IPos + 0, pGLSInfo->IPos + 0, pGLSInfo->DeltPosQc[4] / SQR(pGLSInfo->rmh));
			pGLSInfo->SINSQc.at(pGLSInfo->IPos + 1, pGLSInfo->IPos + 1, pGLSInfo->DeltPosQc[0] / SQR(pGLSInfo->rnh * cos(pGLSInfo->Position[0])));
			pGLSInfo->SINSQc.at(pGLSInfo->IPos + 2, pGLSInfo->IPos + 2, pGLSInfo->DeltPosQc[8]);
		}
	}


	/**
	* @brief       设置SINS-KF的状态变化矩阵CF,仅常数项
	* @param[in]   bReverse     bool
	* @param[in]   pSINSOpt     CSINSOption
	* @param[out]  pGLSInfo     SINS_GLSINFO
	* @return      void
	* @note        松组合/紧组合共用
	* @par History:
	*              2018/01/17,Feng Zhu, new \n
	* @internals
	*/
	void CLooseCouplingBase::setSINSKF_CF_Const(bool bReverse, const CSINSOption* pSINSOpt, SINS_GLSINFO* pGLSInfo)
	{
		if (pSINSOpt->m_NaviSys == NAVI_E)
		{
			double fact = 1.0;
			if (bReverse) fact = -1.0;

			double I[9] = { 0.0 };
			I[0] = I[4] = I[8] = fact;

			double Wie[9] = { 0.0 }; // 这里是正的Wie,下面要的是-Wie
		//	Wie[1] = -gs_WGS84_OMGE;
		//	Wie[3] =  gs_WGS84_OMGE;	

			// 状态矩阵F, 常值部分赋值

			// dPos - dVel
			pGLSInfo->SINSCF.setSub(pGLSInfo->IPos, pGLSInfo->IVel, 3, 3, I);

			// MisAngle - MisAngle
			Wie[1] = gs_WGS84_OMGE * fact;
			Wie[3] = -gs_WGS84_OMGE * fact;
			pGLSInfo->SINSCF.setSub(pGLSInfo->IMisAngle, pGLSInfo->IMisAngle, 3, 3, Wie);

			// dVel - dVel
			Wie[1] = 2.0*gs_WGS84_OMGE*fact;
			Wie[3] = -2.0*gs_WGS84_OMGE*fact;
			pGLSInfo->SINSCF.setSub(pGLSInfo->IVel, pGLSInfo->IVel, 3, 3, Wie);

			// AccelBias - AccelBias
			if (pSINSOpt->m_SSP.PNSD_AccelBiasCTX > IPS_EPSILON) pGLSInfo->SINSCF.at(pGLSInfo->IAccelBias + 0, pGLSInfo->IAccelBias + 0, -1.0 / pSINSOpt->m_SSP.PNSD_AccelBiasCTX);
			if (pSINSOpt->m_SSP.PNSD_AccelBiasCTY > IPS_EPSILON) pGLSInfo->SINSCF.at(pGLSInfo->IAccelBias + 1, pGLSInfo->IAccelBias + 1, -1.0 / pSINSOpt->m_SSP.PNSD_AccelBiasCTY);
			if (pSINSOpt->m_SSP.PNSD_AccelBiasCTZ > IPS_EPSILON) pGLSInfo->SINSCF.at(pGLSInfo->IAccelBias + 2, pGLSInfo->IAccelBias + 2, -1.0 / pSINSOpt->m_SSP.PNSD_AccelBiasCTZ);

			// GyrosBias - GyrosBias
			if (pSINSOpt->m_SSP.PNSD_GyrosBiasCTX > IPS_EPSILON) pGLSInfo->SINSCF.at(pGLSInfo->IGyrosBias + 0, pGLSInfo->IGyrosBias + 0, -1.0 / pSINSOpt->m_SSP.PNSD_GyrosBiasCTX);
			if (pSINSOpt->m_SSP.PNSD_GyrosBiasCTY > IPS_EPSILON) pGLSInfo->SINSCF.at(pGLSInfo->IGyrosBias + 1, pGLSInfo->IGyrosBias + 1, -1.0 / pSINSOpt->m_SSP.PNSD_GyrosBiasCTY);
			if (pSINSOpt->m_SSP.PNSD_GyrosBiasCTZ > IPS_EPSILON) pGLSInfo->SINSCF.at(pGLSInfo->IGyrosBias + 2, pGLSInfo->IGyrosBias + 2, -1.0 / pSINSOpt->m_SSP.PNSD_GyrosBiasCTZ);

			// AccelBiasScale - AccelBiasScale
			if (pSINSOpt->m_bAddAccelBiasScale)
			{
				if (pSINSOpt->m_SSP.PNSD_AccelBiasScaleCTX > IPS_EPSILON) pGLSInfo->SINSCF.at(pGLSInfo->IAccelBiasScale + 0, pGLSInfo->IAccelBiasScale + 0, -1.0 / pSINSOpt->m_SSP.PNSD_AccelBiasScaleCTX);
				if (pSINSOpt->m_SSP.PNSD_AccelBiasScaleCTY > IPS_EPSILON) pGLSInfo->SINSCF.at(pGLSInfo->IAccelBiasScale + 1, pGLSInfo->IAccelBiasScale + 1, -1.0 / pSINSOpt->m_SSP.PNSD_AccelBiasScaleCTY);
				if (pSINSOpt->m_SSP.PNSD_AccelBiasScaleCTZ > IPS_EPSILON) pGLSInfo->SINSCF.at(pGLSInfo->IAccelBiasScale + 2, pGLSInfo->IAccelBiasScale + 2, -1.0 / pSINSOpt->m_SSP.PNSD_AccelBiasScaleCTZ);
			}

			// GyrosBiasScale - GyrosBiasScale
			if (pSINSOpt->m_bAddGyrosBiasScale)
			{
				if (pSINSOpt->m_SSP.PNSD_GyrosBiasScaleCTX > IPS_EPSILON) pGLSInfo->SINSCF.at(pGLSInfo->IGyrosBiasScale + 0, pGLSInfo->IGyrosBiasScale + 0, -1.0 / pSINSOpt->m_SSP.PNSD_GyrosBiasScaleCTX);
				if (pSINSOpt->m_SSP.PNSD_GyrosBiasScaleCTY > IPS_EPSILON) pGLSInfo->SINSCF.at(pGLSInfo->IGyrosBiasScale + 1, pGLSInfo->IGyrosBiasScale + 1, -1.0 / pSINSOpt->m_SSP.PNSD_GyrosBiasScaleCTY);
				if (pSINSOpt->m_SSP.PNSD_GyrosBiasScaleCTZ > IPS_EPSILON) pGLSInfo->SINSCF.at(pGLSInfo->IGyrosBiasScale + 2, pGLSInfo->IGyrosBiasScale + 2, -1.0 / pSINSOpt->m_SSP.PNSD_GyrosBiasScaleCTZ);
			}

			// AccelBiasOrthog - AccelBiasOrthog
			if (pSINSOpt->m_bAddAccelBiasOrthog)
			{
				for (int i = 0; i < 6; i++)
				{
					if (pSINSOpt->m_SSP.PNSD_AccelBiasOrthog[i] > IPS_EPSILON) pGLSInfo->SINSCF.at(pGLSInfo->IAccelBiasOrthog + i, pGLSInfo->IAccelBiasOrthog + i, -1.0 / pSINSOpt->m_SSP.PNSD_AccelBiasOrthog[i]);
				}
			}

			// GyrosBiasOrthog - GyrosBiasOrthog
			if (pSINSOpt->m_bAddGyrosBiasOrthog)
			{
				for (int i = 0; i < 6; i++)
				{
					if (pSINSOpt->m_SSP.PNSD_GyrosBiasOrthog[i] > IPS_EPSILON) pGLSInfo->SINSCF.at(pGLSInfo->IGyrosBiasOrthog + i, pGLSInfo->IGyrosBiasOrthog + i, -1.0 / pSINSOpt->m_SSP.PNSD_GyrosBiasOrthog[i]);
				}
			}
		}
		else if (pSINSOpt->m_NaviSys == NAVI_L)
		{

			// AccelBias - AccelBias
			if (pSINSOpt->m_SSP.PNSD_AccelBiasCTX > IPS_EPSILON) pGLSInfo->SINSCF.at(pGLSInfo->IAccelBias + 0, pGLSInfo->IAccelBias + 0, -1.0 / pSINSOpt->m_SSP.PNSD_AccelBiasCTX);
			if (pSINSOpt->m_SSP.PNSD_AccelBiasCTY > IPS_EPSILON) pGLSInfo->SINSCF.at(pGLSInfo->IAccelBias + 1, pGLSInfo->IAccelBias + 1, -1.0 / pSINSOpt->m_SSP.PNSD_AccelBiasCTY);
			if (pSINSOpt->m_SSP.PNSD_AccelBiasCTZ > IPS_EPSILON) pGLSInfo->SINSCF.at(pGLSInfo->IAccelBias + 2, pGLSInfo->IAccelBias + 2, -1.0 / pSINSOpt->m_SSP.PNSD_AccelBiasCTZ);

			// GyrosBias - GyrosBias
			if (pSINSOpt->m_SSP.PNSD_GyrosBiasCTX > IPS_EPSILON) pGLSInfo->SINSCF.at(pGLSInfo->IGyrosBias + 0, pGLSInfo->IGyrosBias + 0, -1.0 / pSINSOpt->m_SSP.PNSD_GyrosBiasCTX);
			if (pSINSOpt->m_SSP.PNSD_GyrosBiasCTY > IPS_EPSILON) pGLSInfo->SINSCF.at(pGLSInfo->IGyrosBias + 1, pGLSInfo->IGyrosBias + 1, -1.0 / pSINSOpt->m_SSP.PNSD_GyrosBiasCTY);
			if (pSINSOpt->m_SSP.PNSD_GyrosBiasCTZ > IPS_EPSILON) pGLSInfo->SINSCF.at(pGLSInfo->IGyrosBias + 2, pGLSInfo->IGyrosBias + 2, -1.0 / pSINSOpt->m_SSP.PNSD_GyrosBiasCTZ);

			// AccelBiasScale - AccelBiasScale
			if (pSINSOpt->m_bAddAccelBiasScale)
			{
				if (pSINSOpt->m_SSP.PNSD_AccelBiasScaleCTX > IPS_EPSILON) pGLSInfo->SINSCF.at(pGLSInfo->IAccelBiasScale + 0, pGLSInfo->IAccelBiasScale + 0, -1.0 / pSINSOpt->m_SSP.PNSD_AccelBiasScaleCTX);
				if (pSINSOpt->m_SSP.PNSD_AccelBiasScaleCTY > IPS_EPSILON) pGLSInfo->SINSCF.at(pGLSInfo->IAccelBiasScale + 1, pGLSInfo->IAccelBiasScale + 1, -1.0 / pSINSOpt->m_SSP.PNSD_AccelBiasScaleCTY);
				if (pSINSOpt->m_SSP.PNSD_AccelBiasScaleCTZ > IPS_EPSILON) pGLSInfo->SINSCF.at(pGLSInfo->IAccelBiasScale + 2, pGLSInfo->IAccelBiasScale + 2, -1.0 / pSINSOpt->m_SSP.PNSD_AccelBiasScaleCTZ);
			}

			// GyrosBiasScale - GyrosBiasScale
			if (pSINSOpt->m_bAddGyrosBiasScale)
			{
				if (pSINSOpt->m_SSP.PNSD_GyrosBiasScaleCTX > IPS_EPSILON) pGLSInfo->SINSCF.at(pGLSInfo->IGyrosBiasScale + 0, pGLSInfo->IGyrosBiasScale + 0, -1.0 / pSINSOpt->m_SSP.PNSD_GyrosBiasScaleCTX);
				if (pSINSOpt->m_SSP.PNSD_GyrosBiasScaleCTY > IPS_EPSILON) pGLSInfo->SINSCF.at(pGLSInfo->IGyrosBiasScale + 1, pGLSInfo->IGyrosBiasScale + 1, -1.0 / pSINSOpt->m_SSP.PNSD_GyrosBiasScaleCTY);
				if (pSINSOpt->m_SSP.PNSD_GyrosBiasScaleCTZ > IPS_EPSILON) pGLSInfo->SINSCF.at(pGLSInfo->IGyrosBiasScale + 2, pGLSInfo->IGyrosBiasScale + 2, -1.0 / pSINSOpt->m_SSP.PNSD_GyrosBiasScaleCTZ);
			}
		}

	}


	/**
	* @brief       设置SINS-KF的状态变化矩阵CF,仅变化项
	* @param[in]   bReverse     bool
	* @param[in]   pSINSOpt     CSINSOption
	* @param[out]  pGLSInfo     SINS_GLSINFO
	* @return      void
	* @note        松组合/紧组合共用
	* @par History:
	*              2018/01/17,Feng Zhu, new \n
	* @internals
	*/
	void CLooseCouplingBase::setSINSKF_CF_Change(bool bReverse, const CSINSOption* pSINSOpt, SINS_GLSINFO* pGLSInfo)
	{
		if (pSINSOpt->m_NaviSys == NAVI_E)
		{
			// 计算重力张量, 量级较小, 可忽略
			// setSINSKF_TensorOfGrad(pGLSInfo);

			// 得到n系的比力
			double fCross[9];
			double fn[3];
			M33XM31(pGLSInfo->Rbn, pGLSInfo->bAccelCF, fn);
			MatrixSkewSymmetric(fn, fCross);

			pGLSInfo->SINSCF.setSub(pGLSInfo->IVel, pGLSInfo->IPos, 3, 3, pGLSInfo->N_Grav, bReverse); // -m_N
			pGLSInfo->SINSCF.setSub(pGLSInfo->IVel, pGLSInfo->IMisAngle, 3, 3, fCross, bReverse); // -fCross
			pGLSInfo->SINSCF.setSub(pGLSInfo->IVel, pGLSInfo->IAccelBias, 3, 3, pGLSInfo->Rbn, bReverse); // -pGLSInfo->Rbn
			pGLSInfo->SINSCF.setSub(pGLSInfo->IMisAngle, pGLSInfo->IGyrosBias, 3, 3, pGLSInfo->Rbn, !bReverse); // pGLSInfo->Rbn

			if (pSINSOpt->m_bAddAccelBiasScale)
			{
				// dVel - AccelBiasScale
				double Rbef[9];
				Rbef[0] = pGLSInfo->Rbn[0] * pGLSInfo->bAccelCF[0]; Rbef[3] = pGLSInfo->Rbn[3] * pGLSInfo->bAccelCF[0]; Rbef[6] = pGLSInfo->Rbn[6] * pGLSInfo->bAccelCF[0];
				Rbef[1] = pGLSInfo->Rbn[1] * pGLSInfo->bAccelCF[1]; Rbef[4] = pGLSInfo->Rbn[4] * pGLSInfo->bAccelCF[1]; Rbef[7] = pGLSInfo->Rbn[7] * pGLSInfo->bAccelCF[1];
				Rbef[2] = pGLSInfo->Rbn[2] * pGLSInfo->bAccelCF[2]; Rbef[5] = pGLSInfo->Rbn[5] * pGLSInfo->bAccelCF[2]; Rbef[8] = pGLSInfo->Rbn[8] * pGLSInfo->bAccelCF[2];

				//	Rbef[mslice(0,0,3,1)] *= m_bAccelFront(0,0);
				//	Rbef[mslice(0,1,3,1)] *= m_bAccelFront(1,0);
				//	Rbef[mslice(0,2,3,1)] *= m_bAccelFront(2,0);

				pGLSInfo->SINSCF.setSub(pGLSInfo->IVel, pGLSInfo->IAccelBiasScale, 3, 3, Rbef, bReverse);
			}

			if (pSINSOpt->m_bAddGyrosBiasScale)
			{
				// MisAngle - GyrosBiasScale
				//dMatrix Rbew = -pGLSInfo->Rbn;
				//Rbew[mslice(0,0,3,1)] *= m_bGyrosFront(0,0);
				//Rbew[mslice(0,1,3,1)] *= m_bGyrosFront(1,0);
				//Rbew[mslice(0,2,3,1)] *= m_bGyrosFront(2,0);

				double Rbew[9];
				Rbew[0] = -pGLSInfo->Rbn[0] * pGLSInfo->bGyrosCF[0]; Rbew[3] = -pGLSInfo->Rbn[3] * pGLSInfo->bGyrosCF[0]; Rbew[6] = -pGLSInfo->Rbn[6] * pGLSInfo->bGyrosCF[0];
				Rbew[1] = -pGLSInfo->Rbn[1] * pGLSInfo->bGyrosCF[1]; Rbew[4] = -pGLSInfo->Rbn[4] * pGLSInfo->bGyrosCF[1]; Rbew[7] = -pGLSInfo->Rbn[7] * pGLSInfo->bGyrosCF[1];
				Rbew[2] = -pGLSInfo->Rbn[2] * pGLSInfo->bGyrosCF[2]; Rbew[5] = -pGLSInfo->Rbn[5] * pGLSInfo->bGyrosCF[2]; Rbew[8] = -pGLSInfo->Rbn[8] * pGLSInfo->bGyrosCF[2];

				pGLSInfo->SINSCF.setSub(pGLSInfo->IMisAngle, pGLSInfo->IGyrosBiasScale, 3, 3, Rbew, bReverse);
			}

			if (pSINSOpt->m_bAddAccelBiasTob)
			{
				pGLSInfo->SINSCF.setSub(pGLSInfo->IVel, pGLSInfo->IAccelBiasTob, 3, 3, pGLSInfo->Rbn, bReverse); // -pGLSInfo->Rbn
			}
			if (pSINSOpt->m_bAddGyrosBiasTob)
			{
				pGLSInfo->SINSCF.setSub(pGLSInfo->IMisAngle, pGLSInfo->IGyrosBiasTob, 3, 3, pGLSInfo->Rbn, !bReverse); // pGLSInfo->Rbn
			}

			if (pSINSOpt->m_bAddAccelBiasOrthog)
			{
				double RbeM[18];
				double M[18] = { 0.0 };
				M[0 * 6 + 0] = pGLSInfo->bAccelCF[1]; M[0 * 6 + 1] = pGLSInfo->bAccelCF[2];
				M[1 * 6 + 2] = pGLSInfo->bAccelCF[0]; M[1 * 6 + 3] = pGLSInfo->bAccelCF[2];
				M[2 * 6 + 4] = pGLSInfo->bAccelCF[0]; M[2 * 6 + 5] = pGLSInfo->bAccelCF[1];
				MatrixMultiply(3, 3, pGLSInfo->Rbn, 3, 6, M, RbeM);

				pGLSInfo->SINSCF.setSub(pGLSInfo->IVel, pGLSInfo->IAccelBiasOrthog, 3, 6, RbeM, bReverse);
			}

			if (pSINSOpt->m_bAddGyrosBiasOrthog)
			{
				double RbeM[18];
				double M[18] = { 0.0 };
				M[0 * 6 + 0] = pGLSInfo->bGyrosCF[1]; M[0 * 6 + 1] = pGLSInfo->bGyrosCF[2];
				M[1 * 6 + 2] = pGLSInfo->bGyrosCF[0]; M[1 * 6 + 3] = pGLSInfo->bGyrosCF[2];
				M[2 * 6 + 4] = pGLSInfo->bGyrosCF[0]; M[2 * 6 + 5] = pGLSInfo->bGyrosCF[1];
				MatrixMultiply(3, 3, pGLSInfo->Rbn, 3, 6, M, RbeM);

				pGLSInfo->SINSCF.setSub(pGLSInfo->IMisAngle, pGLSInfo->IGyrosBiasOrthog, 3, 6, RbeM, !bReverse);
			}
		}
		else if (pSINSOpt->m_NaviSys == NAVI_L)
		{
			double fact = 1.0;
			if (bReverse) fact = -1.0;

			double FN[3];
			M33XM31(pGLSInfo->Rbn, pGLSInfo->bAccelCF, FN);

			double fE = FN[0];
			double fN = FN[1];
			double fU = FN[2];

			double tF = 0.0;

			double Re = xsqrt(pGLSInfo->rnh*pGLSInfo->rmh);
			double g = pGLSInfo->Gravity[2];

			double we = gs_WGS84_OMGE;

			double vE = pGLSInfo->Velocity[0];
			double vN = pGLSInfo->Velocity[1];
			double vU = pGLSInfo->Velocity[2];

			double cosp = cos(pGLSInfo->Position[0]);
			double sinp = sin(pGLSInfo->Position[0]);
			double tanp = sinp / cosp;

			double dp = vN / pGLSInfo->rmh;
			double dl = vE / (pGLSInfo->rnh*cosp);

			tF = -dp / pGLSInfo->rmh;
			pGLSInfo->SINSCF.at(0, 2, fact*tF);

			tF = 1.0 / pGLSInfo->rmh;
			pGLSInfo->SINSCF.at(0, 4, fact*tF);

			tF = dl * tanp;
			pGLSInfo->SINSCF.at(1, 0, fact*tF);

			tF = -dl / pGLSInfo->rnh;
			pGLSInfo->SINSCF.at(1, 2, fact*tF);

			tF = 1.0 / (pGLSInfo->rnh*cosp);
			pGLSInfo->SINSCF.at(1, 3, fact*tF);

			tF = 1.0;
			pGLSInfo->SINSCF.at(2, 5, fact*tF);

			tF = 2.0 * we * (vU * sinp + vN * cosp) + vN * dl / cosp;
			pGLSInfo->SINSCF.at(3, 0, fact*tF);

			tF = (-vU + dp * tanp*pGLSInfo->rmh) / pGLSInfo->rnh;
			pGLSInfo->SINSCF.at(3, 3, fact*tF);

			tF = (2.0*we + dl)*sinp;
			pGLSInfo->SINSCF.at(3, 4, fact*tF);

			tF = -(2.0*we + dl)*cosp;
			pGLSInfo->SINSCF.at(3, 5, fact*tF);

			tF = -fU;
			pGLSInfo->SINSCF.at(3, 7, fact*tF);

			tF = fN;
			pGLSInfo->SINSCF.at(3, 8, fact*tF);

			tF = -2.0*we*vE*cosp - vE * dl / cosp;
			pGLSInfo->SINSCF.at(4, 0, fact*tF);

			tF = -2.0*(we + dl)*sinp;
			pGLSInfo->SINSCF.at(4, 3, fact*tF);

			tF = -vU / pGLSInfo->rmh;
			pGLSInfo->SINSCF.at(4, 4, fact*tF);

			tF = -dp;
			pGLSInfo->SINSCF.at(4, 5, fact*tF);

			tF = fU;
			pGLSInfo->SINSCF.at(4, 6, fact*tF);

			tF = -fE;
			pGLSInfo->SINSCF.at(4, 8, fact*tF);

			tF = -2.0*we*vE*sinp;
			pGLSInfo->SINSCF.at(5, 0, fact*tF);

			tF = 2.0*g / Re; // 该项影响很小,有时候可以去掉,tF=0
			pGLSInfo->SINSCF.at(5, 2, fact*tF);

			tF = 2.0*(we + dl)*cosp;
			pGLSInfo->SINSCF.at(5, 3, fact*tF);

			tF = 2.0*dp;
			pGLSInfo->SINSCF.at(5, 4, fact*tF);

			tF = -fN;
			pGLSInfo->SINSCF.at(5, 6, fact*tF);

			tF = fE;
			pGLSInfo->SINSCF.at(5, 7, fact*tF);

			tF = dp / pGLSInfo->rmh;
			pGLSInfo->SINSCF.at(6, 2, fact*tF);

			tF = -1.0 / pGLSInfo->rmh;
			pGLSInfo->SINSCF.at(6, 4, fact*tF);

			tF = (we + dl)*sinp;
			pGLSInfo->SINSCF.at(6, 7, fact*tF);

			tF = -(we + dl)*cosp;
			pGLSInfo->SINSCF.at(6, 8, fact*tF);

			tF = -we * sinp;
			pGLSInfo->SINSCF.at(7, 0, fact*tF);

			tF = -dl * cosp / pGLSInfo->rnh;
			pGLSInfo->SINSCF.at(7, 2, fact*tF);

			tF = 1.0 / pGLSInfo->rnh;
			pGLSInfo->SINSCF.at(7, 3, fact*tF);

			tF = -(we + dl)*sinp;
			pGLSInfo->SINSCF.at(7, 6, fact*tF);

			tF = -dp;
			pGLSInfo->SINSCF.at(7, 8, fact*tF);

			tF = we * cosp + dl / cosp;
			pGLSInfo->SINSCF.at(8, 0, fact*tF);

			tF = -dl * sinp / pGLSInfo->rnh;
			pGLSInfo->SINSCF.at(8, 2, fact*tF);

			tF = tanp / pGLSInfo->rnh;
			pGLSInfo->SINSCF.at(8, 3, fact*tF);

			tF = (we + dl)*cosp;
			pGLSInfo->SINSCF.at(8, 6, fact*tF);

			tF = dp;
			pGLSInfo->SINSCF.at(8, 7, fact*tF);

			pGLSInfo->SINSCF.setSub(pGLSInfo->IVel, pGLSInfo->IAccelBias, 3, 3, pGLSInfo->Rbn, bReverse);

			pGLSInfo->SINSCF.setSub(pGLSInfo->IMisAngle, pGLSInfo->IGyrosBias, 3, 3, pGLSInfo->Rbn, !bReverse);


			if (pSINSOpt->m_bAddAccelBiasScale)
			{
				// dVel - AccelBiasScale
				double Rbef[9];
				Rbef[0] = pGLSInfo->Rbn[0] * pGLSInfo->bAccelCF[0]; Rbef[3] = pGLSInfo->Rbn[3] * pGLSInfo->bAccelCF[0]; Rbef[6] = pGLSInfo->Rbn[6] * pGLSInfo->bAccelCF[0];
				Rbef[1] = pGLSInfo->Rbn[1] * pGLSInfo->bAccelCF[1]; Rbef[4] = pGLSInfo->Rbn[4] * pGLSInfo->bAccelCF[1]; Rbef[7] = pGLSInfo->Rbn[7] * pGLSInfo->bAccelCF[1];
				Rbef[2] = pGLSInfo->Rbn[2] * pGLSInfo->bAccelCF[2]; Rbef[5] = pGLSInfo->Rbn[5] * pGLSInfo->bAccelCF[2]; Rbef[8] = pGLSInfo->Rbn[8] * pGLSInfo->bAccelCF[2];

				//	Rbef[mslice(0,0,3,1)] *= m_bAccelFront(0,0);
				//	Rbef[mslice(0,1,3,1)] *= m_bAccelFront(1,0);
				//	Rbef[mslice(0,2,3,1)] *= m_bAccelFront(2,0);

				pGLSInfo->SINSCF.setSub(pGLSInfo->IVel, pGLSInfo->IAccelBiasScale, 3, 3, Rbef, bReverse);
			}

			if (pSINSOpt->m_bAddGyrosBiasScale)
			{
				// MisAngle - GyrosBiasScale
				//dMatrix Rbew = -pGLSInfo->Rbn;
				//Rbew[mslice(0,0,3,1)] *= m_bGyrosFront(0,0);
				//Rbew[mslice(0,1,3,1)] *= m_bGyrosFront(1,0);
				//Rbew[mslice(0,2,3,1)] *= m_bGyrosFront(2,0);

				double Rbew[9];
				Rbew[0] = -pGLSInfo->Rbn[0] * pGLSInfo->bGyrosCF[0]; Rbew[3] = -pGLSInfo->Rbn[3] * pGLSInfo->bGyrosCF[0]; Rbew[6] = -pGLSInfo->Rbn[6] * pGLSInfo->bGyrosCF[0];
				Rbew[1] = -pGLSInfo->Rbn[1] * pGLSInfo->bGyrosCF[1]; Rbew[4] = -pGLSInfo->Rbn[4] * pGLSInfo->bGyrosCF[1]; Rbew[7] = -pGLSInfo->Rbn[7] * pGLSInfo->bGyrosCF[1];
				Rbew[2] = -pGLSInfo->Rbn[2] * pGLSInfo->bGyrosCF[2]; Rbew[5] = -pGLSInfo->Rbn[5] * pGLSInfo->bGyrosCF[2]; Rbew[8] = -pGLSInfo->Rbn[8] * pGLSInfo->bGyrosCF[2];

				pGLSInfo->SINSCF.setSub(pGLSInfo->IMisAngle, pGLSInfo->IGyrosBiasScale, 3, 3, Rbew, bReverse);
			}

			if (pSINSOpt->m_bAddAccelBiasTob)
			{
				// dVel - AccelBiasTob
				pGLSInfo->SINSCF.setSub(pGLSInfo->IVel, pGLSInfo->IAccelBiasTob, 3, 3, pGLSInfo->Rbn, bReverse);
			}
			if (pSINSOpt->m_bAddGyrosBiasTob)
			{
				// MisAngle - GyrosBiasTob
				pGLSInfo->SINSCF.setSub(pGLSInfo->IMisAngle, pGLSInfo->IGyrosBiasTob, 3, 3, pGLSInfo->Rbn, !bReverse);
			}

			if (pSINSOpt->m_bAddAccelBiasOrthog)
			{
				double RbeM[18];
				double M[18] = { 0.0 };
				M[0 * 6 + 0] = pGLSInfo->bAccelCF[1]; M[0 * 6 + 1] = pGLSInfo->bAccelCF[2];
				M[1 * 6 + 2] = pGLSInfo->bAccelCF[0]; M[1 * 6 + 3] = pGLSInfo->bAccelCF[2];
				M[2 * 6 + 4] = pGLSInfo->bAccelCF[0]; M[2 * 6 + 5] = pGLSInfo->bAccelCF[1];
				MatrixMultiply(3, 3, pGLSInfo->Rbn, 3, 6, M, RbeM);

				pGLSInfo->SINSCF.setSub(pGLSInfo->IVel, pGLSInfo->IAccelBiasOrthog, 3, 6, RbeM, bReverse);
			}

			if (pSINSOpt->m_bAddGyrosBiasOrthog)
			{
				double RbeM[18];
				double M[18] = { 0.0 };
				M[0 * 6 + 0] = pGLSInfo->bGyrosCF[1]; M[0 * 6 + 1] = pGLSInfo->bGyrosCF[2];
				M[1 * 6 + 2] = pGLSInfo->bGyrosCF[0]; M[1 * 6 + 3] = pGLSInfo->bGyrosCF[2];
				M[2 * 6 + 4] = pGLSInfo->bGyrosCF[0]; M[2 * 6 + 5] = pGLSInfo->bGyrosCF[1];
				MatrixMultiply(3, 3, pGLSInfo->Rbn, 3, 6, M, RbeM);

				pGLSInfo->SINSCF.setSub(pGLSInfo->IMisAngle, pGLSInfo->IGyrosBiasOrthog, 3, 6, RbeM, !bReverse);
			}
		}
	}


	/**
	* @brief       进行SINS状态的闭环修正
	* @param[in]   pSINSOpt     CSINSOption
	* @param[in]   pSINSSol     CSINSSolution
	* @param[out]  pGLSInfo     SINS_GLSINFO
	* @param[out]  pIMUFFStream CIMUFFStream
	* @return      void
	* @note        松组合/紧组合共用
	* @par History:
	*              2018/01/17,Feng Zhu, new \n
	* @internals
	*/
	void CLooseCouplingBase::setSINSKF_ClosedLoop(const CSINSOption* pSINSOpt, SINS_GLSINFO* pGLSInfo, CIMUFFStream* pIMUFFStream)
	{
		// 状态量初始化为0，方差不变
		for (int i = 0; i < 3; i++)
		{
			// 零偏反馈到IMU观测值进行修正
			pIMUFFStream->m_bBiasFedBack = pSINSOpt->m_bBiasFedBack;
			pIMUFFStream->m_AccelBias[i] = pGLSInfo->AccelBias[i];
			pIMUFFStream->m_GyrosBias[i] = pGLSInfo->GyrosBias[i];
			if (pSINSOpt->m_bAddAccelBiasScale) pIMUFFStream->m_AccelBiasScale[i] = pGLSInfo->AccelBiasScale[i];
			if (pSINSOpt->m_bAddGyrosBiasScale) pIMUFFStream->m_GyrosBiasScale[i] = pGLSInfo->GyrosBiasScale[i];
			if (pSINSOpt->m_bAddAccelBiasTob) pIMUFFStream->m_AccelBiasTob[i] = pGLSInfo->AccelBiasTob[i];
			if (pSINSOpt->m_bAddGyrosBiasTob) pIMUFFStream->m_GyrosBiasTob[i] = pGLSInfo->GyrosBiasTob[i];

			if (pSINSOpt->m_bAddAccelBiasOrthog)
			{
				pIMUFFStream->m_AccelBiasOrthog[i] = pGLSInfo->AccelBiasOrthog[i];
				pIMUFFStream->m_AccelBiasOrthog[i + 3] = pGLSInfo->AccelBiasOrthog[i + 3];
			}
			if (pSINSOpt->m_bAddGyrosBiasOrthog)
			{
				pIMUFFStream->m_GyrosBiasOrthog[i] = pGLSInfo->GyrosBiasOrthog[i];
				pIMUFFStream->m_GyrosBiasOrthog[i + 3] = pGLSInfo->GyrosBiasOrthog[i + 3];
			}

			pGLSInfo->StateX.at(pGLSInfo->IPos, i, 0.0);
			pGLSInfo->StateX.at(pGLSInfo->IVel, i, 0.0);
			if (pSINSOpt->m_bAttFedBack) pGLSInfo->StateX.at(pGLSInfo->IMisAngle, i, 0.0);

			// 零偏初始化
			if (pSINSOpt->m_bBiasFedBack)
			{
				pGLSInfo->StateX.at(pGLSInfo->IAccelBias, i, 0.0);
				pGLSInfo->StateX.at(pGLSInfo->IGyrosBias, i, 0.0);

				if (pSINSOpt->m_bAddAccelBiasScale) pGLSInfo->StateX.at(pGLSInfo->IAccelBiasScale, i, 0.0);
				if (pSINSOpt->m_bAddGyrosBiasScale) pGLSInfo->StateX.at(pGLSInfo->IGyrosBiasScale, i, 0.0);

				if (pSINSOpt->m_bAddAccelBiasTob) pGLSInfo->StateX.at(pGLSInfo->IAccelBiasTob, i, 0.0);
				if (pSINSOpt->m_bAddGyrosBiasTob) pGLSInfo->StateX.at(pGLSInfo->IGyrosBiasTob, i, 0.0);

				if (pSINSOpt->m_bAddAccelBiasOrthog)
				{
					pGLSInfo->StateX.at(pGLSInfo->IAccelBiasOrthog, i, 0.0);
					pGLSInfo->StateX.at(pGLSInfo->IAccelBiasOrthog, i + 3, 0.0);
				}
				if (pSINSOpt->m_bAddGyrosBiasOrthog)
				{
					pGLSInfo->StateX.at(pGLSInfo->IGyrosBiasOrthog, i, 0.0);
					pGLSInfo->StateX.at(pGLSInfo->IGyrosBiasOrthog, i + 3, 0.0);
				}
			}
		}
	}


	/**
	* @brief       ECEF系下得到SINS的重力张量
	* @param[out]  pGLSInfo     SINS_GLSINFO
	* @return      void
	* @note        松组合/紧组合共用
	* @par History:
	*              2018/01/17,Feng Zhu, new \n
	* @internals
	*/
	void CLooseCouplingBase::setSINSKF_TensorOfGrad(SINS_GLSINFO* pGLSInfo)
	{
		int i, j;
		double r = MatrixNorm2(3, 1, pGLSInfo->Position);
		double r2 = r * r;
		double ws2 = gs_WGS84_GME / (r2*r);

		//m_N.unit();
		//m_N -= 3.0 * (m_GLSInfo.Position * ~m_GLSInfo.Position) / (r2);
		//m_N *= (-ws2);

		MatrixMultiply_HPHT(3, 1, pGLSInfo->Position, NULL, false, pGLSInfo->N_Grav);

		for (i = 0; i < 3; i++)
		{
			for (j = 0; j < 3; j++)
			{
				pGLSInfo->N_Grav[i * 3 + j] *= (3.0 / r2);

				if (i == j)
				{
					pGLSInfo->N_Grav[i * 3 + j] = 1.0 - pGLSInfo->N_Grav[i * 3 + j];
				}
				else
				{
					pGLSInfo->N_Grav[i * 3 + j] *= -1.0;
				}
			}
		}

		M33Scale(-ws2, pGLSInfo->N_Grav);
		pGLSInfo->N_Grav[0] += SQR(gs_WGS84_OMGE);
		pGLSInfo->N_Grav[4] += SQR(gs_WGS84_OMGE);
	}
	

	/**
	* @brief       SINS中获得地球重力和自转参数
	* @param[in]   pSINSOpt     CSINSOption
	* @param[out]  pGLSInfo     SINS_GLSINFO
	* @param[in]   pSimpleEGM   CSimpleEGM
	* @param[in]   Pos          double
	* @param[in]   Vel          double
	* @param[in]   bPos         bool
	* @param[in]   bVel         bool
	* @return      void
	* @note        松组合/紧组合共用
	* @par History:
	*              2018/01/17,Feng Zhu, new \n
	* @internals
	*/
	void CLooseCouplingBase::setSINSKF_EarthPara(const CSINSOption* pSINSOpt, SINS_GLSINFO* pGLSInfo, CSimpleEGM* pSimpleEGM, double Pos[3], double Vel[3], bool bPos, bool bVel)
	{
		if (pSINSOpt->m_NaviSys == NAVI_E)
		{
			if (bPos)
			{
				pSimpleEGM->setXYZ(Pos);
				pSimpleEGM->getGravityECEF(pGLSInfo->Gravity);
			}
		}
		else if (pSINSOpt->m_NaviSys == NAVI_L)
		{
			if (bPos)
			{
				pSimpleEGM->setLLH(Pos);
				pGLSInfo->Gravity[2] = -pSimpleEGM->getGravityLocal();
			}

			pSimpleEGM->getEarthPara(Vel, pGLSInfo->Wnie, pGLSInfo->Wnen, &pGLSInfo->rmh, &pGLSInfo->rnh, bVel);
		}
	}








	// ************************************************************************************************************************************** //
	//  previous class : CLooseCouplingBase ;     next class : CLooseCouplingECEF                                                             //
	// ************************************************************************************************************************************** //









	CLooseCouplingECEF::CLooseCouplingECEF()
	{
		
	}

	CLooseCouplingECEF::~CLooseCouplingECEF()
	{

	}


	bool CLooseCouplingECEF::initLC(bool bReverse)
	{
		m_bReverse = bReverse;
		m_pSINSOpt->m_NaviSys = NAVI_E;
		return initLCBase();
	}


	bool CLooseCouplingECEF::setMEQ(const int iter, const int ECLType)
	{
		ELC_DATA* pELC = &m_pELCFFStream->m_ELCData;
		bool bMeasPOS = m_GLSInfo.bMeasPOS&&(ECLType&ELCTYPE_POS) && MatrixTrace(3,3,m_pELCFFStream->m_ELCData.GNSS_DATA.XYZPosP) > 0.0;
		bool bMeasVEL = m_GLSInfo.bMeasVEL&&(ECLType&ELCTYPE_VEL) && MatrixTrace(3,3,m_pELCFFStream->m_ELCData.GNSS_DATA.XYZVelP) > 0.0;
		bool bMeasATT = m_GLSInfo.bMeasATT&&(ECLType&ELCTYPE_ATT) && MatrixTrace(3,3,m_pELCFFStream->m_ELCData.GNSS_DATA.AttP) > 0.0;
		bool bMeasHED = m_GLSInfo.bMeasHED&&(ECLType&ELCTYPE_HED) && m_pELCFFStream->m_ELCData.HED_DATA.HeadingP > 0.0;
		bool bMeasDMI = m_GLSInfo.bMeasDMI&&(ECLType&ELCTYPE_DMI) && m_pELCFFStream->m_ELCData.DMI_DATA.DMIVelP  > 0.0;
		bool bMeasZUP = m_GLSInfo.bMeasZUP&&(ECLType&ELCTYPE_ZUP);
		bool bMeasZIH = m_GLSInfo.bMeasZIH&&(ECLType&ELCTYPE_ZIH);
		bool bMeasCUP = m_GLSInfo.bMeasCUP&&(ECLType&ELCTYPE_CUP) && MatrixTrace(3,3,m_pELCFFStream->m_ELCData.CUP_DATA.CUPTPosP) > 0.0;
		bool bMeasNHC = m_GLSInfo.bMeasNHC&&(ECLType&ELCTYPE_NHC);
		bool bMeasHGH = m_GLSInfo.bMeasHGH&&(ECLType&ELCTYPE_HGH);
		bMeasHGH = false; // 高程和垂向速度约束应该等价,作用不大

		//XZ 模拟失锁
		//if (pELC->gt.secsOfWeek >= 113390 && pELC->gt.secsOfWeek <= 113690)
		//{
		//	bMeasPOS = false; bMeasVEL = false; bMeasATT = false;
		//	bMeasHED = false; bMeasDMI = false; bMeasZUP = false;
		//	bMeasZIH = false; bMeasCUP = false; bMeasNHC = false;
		//	bMeasHGH = false;
		//}

		//直线模拟失锁，检查惯导递推的发散情况
		//if (pELC->gt.secsOfWeek >= 120492 && pELC->gt.secsOfWeek <= 120552)
		//{
		//	bMeasPOS = false; bMeasVEL = false; bMeasATT = false;
		//	bMeasHED = false; bMeasDMI = false; bMeasZUP = false;
		//	bMeasZIH = false; bMeasCUP = false; bMeasNHC = false;
		//	bMeasHGH = false;
		//}

		if (pELC->gt.secsOfWeek >= 120933)
		{
			double asdas = 0;
		}


		// 观测更新率判断
		if (bMeasPOS && m_pSINSOpt->m_POSRate > 0)
		{
			if (m_bReverse)
			{
				if (m_UpdateTime <= m_MeasPOSTime0 || m_MeasPOSTime0.GPSWeek < 0)
				{
					if (m_MeasPOSTime0.GPSWeek < 0) m_MeasPOSTime0 = m_UpdateTime;
					m_MeasPOSTime0 -= m_pSINSOpt->m_POSRate;
					bMeasPOS = true;
				}
				else
				{
					bMeasPOS = false;
				}
			}
			else
			{
				if (m_UpdateTime >= m_MeasPOSTime0 || m_MeasPOSTime0.GPSWeek < 0)
				{
					if (m_MeasPOSTime0.GPSWeek < 0) m_MeasPOSTime0 = m_UpdateTime;
					m_MeasPOSTime0 += m_pSINSOpt->m_POSRate;
					bMeasPOS = true;
				}
				else
				{
					bMeasPOS = false;
				}
			}	
		}

		// 满足以下条件,说明位置解算的不好,直接不用位置
		if (bMeasPOS)
		{
			// 半紧组合不做以下判断,因为卫星数很少的时候,尤其小于4颗,根本无法计算PDOP值
			if (m_bSTC == false)
			{
				if (m_pELCFFStream->m_ELCData.GNSS_DATA.DDOP >= 25.0) bMeasPOS = false;
				if (MatrixTrace(3, 3, m_pELCFFStream->m_ELCData.GNSS_DATA.XYZPosP) >= 30.0) bMeasPOS = false;
				if (MatrixNorm2(3, 3, m_pELCFFStream->m_ELCData.GNSS_DATA.SepENU) > 30.0) bMeasPOS = false;
				//if (m_pELCFFStream->m_ELCData.GNSS_DATA.Quality > 5) bMeasPOS = false;
				//if (m_pELCFFStream->m_ELCData.GNSS_DATA.AmbState < 2) bMeasPOS = false;
				//if (m_pELCFFStream->m_ELCData.GNSS_DATA.ARRatio < 3.5) bMeasPOS = false;
			}
		}

		// 航向容易估计错误，这时候还不如不用
		if ((fabs(m_pSINSOpt->m_RotAngleNHC[0]) > 2 || fabs(m_pSINSOpt->m_RotAngleNHC[1]) > 2) && bMeasNHC)
		{
			bMeasNHC = false;
		}

		// 没有估计里程计误差，就先不做DMI+NHC
		if ((gs_bHCDH || gs_bLJYY) && m_pSINSOpt->m_dWheelSizeDMI == 0 && bMeasDMI)
		{
			bMeasDMI = false;
		}
		// 没有估计安置角误差，就先不做NHC
		if ((gs_bHCDH || gs_bLJYY) && m_pSINSOpt->m_RotAngleNHC[0] == 0 && m_pSINSOpt->m_RotAngleNHC[1] == 0 && bMeasNHC)
		{
			bMeasNHC = false;
		}
		// 有里程计的情况下，大幅提高位置拒绝的阈值
		else if((gs_bHCDH || gs_bLJYY) && m_pSINSOpt->m_dWheelSizeDMI != 0 && bMeasPOS)
		{
			///< ToDo(IPS2.3): 
			// 前后向互差较大或不是固定解拒绝位置
			if (MatrixNorm2(3, 1, m_pELCFFStream->m_ELCData.GNSS_DATA.SepENU) > 3.0 || 
				m_pELCFFStream->m_ELCData.GNSS_DATA.AmbState < 5 || 
				m_pELCFFStream->m_ELCData.GNSS_DATA.Quality > 2)
			{
				bMeasPOS = false;

				// GNSS偶尔可能前向是对的，后向是错的并且差了十米，而双向结果采纳了前向的，因此用前后向互差必须慎重
				if (m_pELCFFStream->m_ELCData.GNSS_DATA.ARRatio > 10.0) bMeasPOS = true;
				if (m_pELCFFStream->m_ELCData.GNSS_DATA.DDOP < 2.0 && m_pELCFFStream->m_ELCData.GNSS_DATA.DDOP > 0.1) bMeasPOS = true;
				if (MatrixTrace(3, 3, m_pELCFFStream->m_ELCData.GNSS_DATA.XYZPosP) < 2.0 &&
					m_pELCFFStream->m_ELCData.GNSS_DATA.ARRatio > 3.0) bMeasPOS = true;
			}
		}
		// 没有VEL或者DMI，不单独做NHC
		if (bMeasNHC && !bMeasDMI)
		{
			bMeasNHC = false;
		}
		
		// 所有观测更新均为false,则直接返回
		if ((!bMeasPOS) && (!bMeasVEL) && (!bMeasATT) && (!bMeasHED) && (!bMeasDMI) &&
		   (!bMeasZUP)  && (!bMeasZIH) && (!bMeasCUP) && (!bMeasNHC) && (!bMeasHGH))
		   return false;


		// 1. 位置、速度、姿态修正,得到较为准确的值,减弱观测方程非线性
		M31_M31(m_SINSMech.m_Position, m_GLSInfo.StateX.x + m_GLSInfo.IPos, m_GLSInfo.Position);
		M31_M31(m_SINSMech.m_Velocity, m_GLSInfo.StateX.x + m_GLSInfo.IVel, m_GLSInfo.Velocity);
		CorrectMisAngle(m_SINSMech.m_Rbn, m_GLSInfo.StateX.x + m_GLSInfo.IMisAngle, m_GLSInfo.Rbn);
		UnitOrthoRotMatrix(m_GLSInfo.Rbn);

		// 2. m_MeasureHMatrix 等清零
		m_GLSInfo.MeasH_ELC.zero();
		m_GLSInfo.MeasR_ELC.zero();
		m_GLSInfo.MeasInno_ELC.zero();
		m_GLSInfo.nMeas_ELC = 0;

		double M1[3], M2[3], R1[9];

		double nlarm[3];
		M33XM31(m_GLSInfo.Rbn, m_pSINSOpt->m_LeverArmbGNSS, nlarm);
		double nlarmX[9];
		MatrixSkewSymmetric(nlarm, nlarmX);

		if(bMeasPOS)
		{
			m_GLSInfo.MeasH_ELC.at(m_GLSInfo.nMeas_ELC + 0, 0, 1.0);
			m_GLSInfo.MeasH_ELC.at(m_GLSInfo.nMeas_ELC + 1, 1, 1.0);
			m_GLSInfo.MeasH_ELC.at(m_GLSInfo.nMeas_ELC + 2, 2, 1.0);
			m_GLSInfo.MeasH_ELC.setSub(0, m_GLSInfo.IMisAngle, 3, 3, nlarmX);

			M31M31(m_GLSInfo.Position, nlarm, M1);
			M31_M31(M1, pELC->GNSS_DATA.XYZPos, M2);
			m_GLSInfo.MeasInno_ELC.setSub(0, 0, 3, 1, M2);

		//	double GPSP[9];
		//	M33EQU(pELC->GPSPosP, GPSP); M33Scale(25.0,GPSP);
		//	m_GLSInfo.MeasR_ELC.setSub(0, 0, 3, 3, GPSP);

			m_GLSInfo.MeasR_ELC.setSub(0, 0, 3, 3, pELC->GNSS_DATA.XYZPosP);
			m_GLSInfo.nMeas_ELC += NMeas_POS;		

			// 后面越迭代越小, 主要看第一次
			if (iter == 1) M31EQU(M2, m_pSINSSol->m_MisclosurePos);
		}


		if (bMeasVEL || bMeasZUP)
		{
			double blarmX[9];
			MatrixSkewSymmetric(m_pSINSOpt->m_LeverArmbGNSS, blarmX);

			double wie[3] = { 0.0 };
			wie[2] = gs_WGS84_OMGE;
			// dMatrix larmw  = Cross3(nlarm,m_pIMUFFStream->m_bGyros) + Cross3(wie,nlarm); // 容易犯错误,TC中影响很明显
			// dMatrix larmw  = Cross3(wie,nlarm) + m_GLSInfo.Rbn*Cross3(m_LeverArmbGNSS0,m_pIMUFFStream->m_bGyros);
			double larmw[3] = { 0.0 };
			CrossM3(wie, nlarm, M1);
			CrossM3(m_pSINSOpt->m_LeverArmbGNSS, m_pIMUFFStream->m_bGyros, M2);
			M33XM31(m_GLSInfo.Rbn, M2, larmw);
			M31M31(M1, larmw);

			double larmwX[9];
			MatrixSkewSymmetric(larmw, larmwX);

			m_GLSInfo.MeasH_ELC.at(m_GLSInfo.nMeas_ELC + 0, 3, 1.0);
			m_GLSInfo.MeasH_ELC.at(m_GLSInfo.nMeas_ELC + 1, 4, 1.0);
			m_GLSInfo.MeasH_ELC.at(m_GLSInfo.nMeas_ELC + 2, 5, 1.0);

			// 速度更新的H矩阵
			if(bMeasZUP == false)
			{
				m_GLSInfo.MeasH_ELC.setSub(m_GLSInfo.nMeas_ELC, m_GLSInfo.IMisAngle, 3, 3, larmwX, true);
				M33XM33(m_GLSInfo.Rbn, blarmX, R1);
				m_GLSInfo.MeasH_ELC.setSub(m_GLSInfo.nMeas_ELC, m_GLSInfo.IGyrosBias, 3, 3, R1, true);
				
				if (m_pSINSOpt->m_bAddGyrosBiasTob)
				{
					m_GLSInfo.MeasH_ELC.setSub(m_GLSInfo.nMeas_ELC, m_GLSInfo.IGyrosBiasTob, 3, 3, R1, true);
				}

				if(m_pSINSOpt->m_bAddGyrosBiasScale)
				{
					M33XM33(m_GLSInfo.Rbn, blarmX, R1);
					R1[0] *= m_pIMUFFStream->m_bGyros[0]; R1[3] *= m_pIMUFFStream->m_bGyros[0]; R1[6] *= m_pIMUFFStream->m_bGyros[0];
					R1[1] *= m_pIMUFFStream->m_bGyros[1]; R1[4] *= m_pIMUFFStream->m_bGyros[1]; R1[7] *= m_pIMUFFStream->m_bGyros[1];
					R1[2] *= m_pIMUFFStream->m_bGyros[2]; R1[5] *= m_pIMUFFStream->m_bGyros[2]; R1[8] *= m_pIMUFFStream->m_bGyros[2];
					m_GLSInfo.MeasH_ELC.setSub(m_GLSInfo.nMeas_ELC, m_GLSInfo.IGyrosBiasScale, 3, 3, R1, true);
				}

				if (m_pSINSOpt->m_bAddGyrosBiasOrthog)
				{
					M33XM33(m_GLSInfo.Rbn, blarmX, R1);

					double RbeM[18];
					double M[18] = { 0.0 };
					M[0 * 6 + 0] = m_pIMUFFStream->m_bGyros[1]; M[0 * 6 + 1] = m_pIMUFFStream->m_bGyros[2];
					M[1 * 6 + 2] = m_pIMUFFStream->m_bGyros[0]; M[1 * 6 + 3] = m_pIMUFFStream->m_bGyros[2];
					M[2 * 6 + 4] = m_pIMUFFStream->m_bGyros[0]; M[2 * 6 + 5] = m_pIMUFFStream->m_bGyros[1];
					MatrixMultiply(3, 3, R1, 3, 6, M, RbeM);

					m_GLSInfo.MeasH_ELC.setSub(m_GLSInfo.nMeas_ELC, m_GLSInfo.IGyrosBiasOrthog, 3, 6, RbeM, true);
				}
			}

			// 零速更新
			if(bMeasZUP)
			{
				double ZUPVar = SQR(0.0175);

				// 推算时间一长就把零速放宽点
				if (m_PosUpt_Pre.GPSWeek != -1)
				{
					if (fabs(m_UpdateTime - m_PosUpt_Pre) > 120)
					{
						ZUPVar = SQR(0.2);
					}
					else if (fabs(m_UpdateTime - m_PosUpt_Pre) > 60)
					{
						ZUPVar = SQR(0.15);
					}
					else if (fabs(m_UpdateTime - m_PosUpt_Pre) > 30)
					{
						ZUPVar = SQR(0.1);
					}

				}

				m_GLSInfo.MeasInno_ELC.setSub(m_GLSInfo.nMeas_ELC, 0, 3, 1, m_GLSInfo.Velocity, false);
				m_GLSInfo.MeasR_ELC.at(m_GLSInfo.nMeas_ELC + 0, m_GLSInfo.nMeas_ELC + 0, ZUPVar);
				m_GLSInfo.MeasR_ELC.at(m_GLSInfo.nMeas_ELC + 1, m_GLSInfo.nMeas_ELC + 1, ZUPVar);
				m_GLSInfo.MeasR_ELC.at(m_GLSInfo.nMeas_ELC + 2, m_GLSInfo.nMeas_ELC + 2, ZUPVar);
				
			}
			else
			{				
				M31_M31(m_GLSInfo.Velocity, larmw, M1);
				M31_M31(M1, pELC->GNSS_DATA.XYZVel, M2);
				m_GLSInfo.MeasInno_ELC.setSub(m_GLSInfo.nMeas_ELC, 0, 3, 1, M2, false);
				m_GLSInfo.MeasR_ELC.setSub(m_GLSInfo.nMeas_ELC, m_GLSInfo.nMeas_ELC, 3, 3, pELC->GNSS_DATA.XYZVelP);
			}
			
			m_GLSInfo.nMeas_ELC += NMeas_VEL;
		}


		if(bMeasDMI)
		{
			m_GLSInfo.bEstDMIScale = false; 
			m_GLSInfo.DMIScale = 1.0 / (1.0 + m_pSINSOpt->m_dWheelSizeDMI);

			if(pELC->DMI_DATA.DMIVel > 0.01)
			{
				MatrixTranspose(3, 3, m_GLSInfo.Rbn, R1);
				M33XM31(R1, m_GLSInfo.Velocity, M1);

				m_GLSInfo.MeasH_ELC.at(m_GLSInfo.nMeas_ELC, m_GLSInfo.IVel + 0, m_GLSInfo.Rbn[1]);
				m_GLSInfo.MeasH_ELC.at(m_GLSInfo.nMeas_ELC, m_GLSInfo.IVel + 1, m_GLSInfo.Rbn[4]);
				m_GLSInfo.MeasH_ELC.at(m_GLSInfo.nMeas_ELC, m_GLSInfo.IVel + 2, m_GLSInfo.Rbn[7]);

				m_GLSInfo.MeasH_ELC.at(m_GLSInfo.nMeas_ELC, m_GLSInfo.IMisAngle + 0, (m_GLSInfo.Rbn[5] * gs_WGS84_OMGE*m_pSINSOpt->m_LeverArmDMI[0] - m_GLSInfo.Rbn[1] * gs_WGS84_OMGE*m_pSINSOpt->m_LeverArmDMI[2] + M1[2]));
				m_GLSInfo.MeasH_ELC.at(m_GLSInfo.nMeas_ELC, m_GLSInfo.IMisAngle + 1, (m_GLSInfo.Rbn[0] * gs_WGS84_OMGE*m_pSINSOpt->m_LeverArmDMI[2] - m_GLSInfo.Rbn[2] * gs_WGS84_OMGE*m_pSINSOpt->m_LeverArmDMI[0]));
				m_GLSInfo.MeasH_ELC.at(m_GLSInfo.nMeas_ELC, m_GLSInfo.IMisAngle + 2, (M1[0]));

				m_GLSInfo.MeasH_ELC.at(m_GLSInfo.nMeas_ELC, m_GLSInfo.IGyrosBias + 0, (-m_pSINSOpt->m_LeverArmDMI[2]));
				m_GLSInfo.MeasH_ELC.at(m_GLSInfo.nMeas_ELC, m_GLSInfo.IGyrosBias + 2, (m_pSINSOpt->m_LeverArmDMI[0]));

				if(m_pSINSOpt->m_bAddGyrosBiasScale)
				{
					m_GLSInfo.MeasH_ELC.at(m_GLSInfo.nMeas_ELC, m_GLSInfo.IGyrosBiasScale + 0, (-m_pSINSOpt->m_LeverArmDMI[2] * m_pIMUFFStream->m_bGyros[0]));
					m_GLSInfo.MeasH_ELC.at(m_GLSInfo.nMeas_ELC, m_GLSInfo.IGyrosBiasScale + 2, (m_pSINSOpt->m_LeverArmDMI[0] * m_pIMUFFStream->m_bGyros[2]));
				}

				if(m_pSINSOpt->m_bAddGyrosBiasTob)
				{
					m_GLSInfo.MeasH_ELC.at(m_GLSInfo.nMeas_ELC, m_GLSInfo.IGyrosBiasTob + 0, (-m_pSINSOpt->m_LeverArmDMI[2]));
					m_GLSInfo.MeasH_ELC.at(m_GLSInfo.nMeas_ELC, m_GLSInfo.IGyrosBiasTob + 2, ( m_pSINSOpt->m_LeverArmDMI[0]));
				}

				if (m_pSINSOpt->m_bAddGyrosBiasOrthog)
				{
					///< ToDo(IPS2.2): 
				}

				double meas = M1[1] - m_GLSInfo.DMIScale*pELC->DMI_DATA.DMIVel;
				meas += (m_GLSInfo.Rbn[6] * gs_WGS84_OMGE*m_pSINSOpt->m_LeverArmDMI[2] - m_GLSInfo.Rbn[8] * gs_WGS84_OMGE*m_pSINSOpt->m_LeverArmDMI[0]);
				meas += (m_pIMUFFStream->m_bGyros[2] * m_pSINSOpt->m_LeverArmDMI[0] - m_pIMUFFStream->m_bGyros[0] * m_pSINSOpt->m_LeverArmDMI[2]);
				
				m_GLSInfo.MeasInno_ELC.at(m_GLSInfo.nMeas_ELC, 0, meas);
				m_GLSInfo.MeasR_ELC.at(m_GLSInfo.nMeas_ELC, m_GLSInfo.nMeas_ELC, SQR(0.35));

				m_GLSInfo.nMeas_ELC += NMeas_DMI;
			}
		}


		if(bMeasNHC)
		{
			if(_FRAC((m_pELCFFStream->m_time - m_UpdateTime)/m_pELCFFStream->m_DeltT) == 0.0)
			{
				m_GLSInfo.MeasH_ELC.at(m_GLSInfo.nMeas_ELC + 0, 3, m_GLSInfo.Rbn[0]);
				m_GLSInfo.MeasH_ELC.at(m_GLSInfo.nMeas_ELC + 0, 4, m_GLSInfo.Rbn[3]);
				m_GLSInfo.MeasH_ELC.at(m_GLSInfo.nMeas_ELC + 0, 5, m_GLSInfo.Rbn[6]);
				m_GLSInfo.MeasH_ELC.at(m_GLSInfo.nMeas_ELC + 0, 6, (m_GLSInfo.Rbn[6] * m_GLSInfo.Velocity[1] - m_GLSInfo.Rbn[3] * m_GLSInfo.Velocity[2]));
				m_GLSInfo.MeasH_ELC.at(m_GLSInfo.nMeas_ELC + 0, 7, (m_GLSInfo.Rbn[0] * m_GLSInfo.Velocity[2] - m_GLSInfo.Rbn[6] * m_GLSInfo.Velocity[0]));
				m_GLSInfo.MeasH_ELC.at(m_GLSInfo.nMeas_ELC + 0, 8, (m_GLSInfo.Rbn[3] * m_GLSInfo.Velocity[0] - m_GLSInfo.Rbn[0] * m_GLSInfo.Velocity[1]));

				m_GLSInfo.MeasH_ELC.at(m_GLSInfo.nMeas_ELC + 1, 3, m_GLSInfo.Rbn[2]);
				m_GLSInfo.MeasH_ELC.at(m_GLSInfo.nMeas_ELC + 1, 4, m_GLSInfo.Rbn[5]);
				m_GLSInfo.MeasH_ELC.at(m_GLSInfo.nMeas_ELC + 1, 5, m_GLSInfo.Rbn[8]);
				m_GLSInfo.MeasH_ELC.at(m_GLSInfo.nMeas_ELC + 1, 6, (m_GLSInfo.Rbn[8] * m_GLSInfo.Velocity[1] - m_GLSInfo.Rbn[5] * m_GLSInfo.Velocity[2]));
				m_GLSInfo.MeasH_ELC.at(m_GLSInfo.nMeas_ELC + 1, 7, (m_GLSInfo.Rbn[2] * m_GLSInfo.Velocity[2] - m_GLSInfo.Rbn[8] * m_GLSInfo.Velocity[0]));
				m_GLSInfo.MeasH_ELC.at(m_GLSInfo.nMeas_ELC + 1, 8, (m_GLSInfo.Rbn[5] * m_GLSInfo.Velocity[0] - m_GLSInfo.Rbn[2] * m_GLSInfo.Velocity[1]));

				MatrixTranspose(3, 3, m_GLSInfo.Rbn, R1);
				M33XM31(R1, m_GLSInfo.Velocity, M1);

				// 取出XZ方向的速度作为观测	
				//m_GLSInfo.MeasInno_ELC.at(m_GLSInfo.nMeas_ELC    , 0, M1[0]);
				//m_GLSInfo.MeasInno_ELC.at(m_GLSInfo.nMeas_ELC + 1, 0, M1[2]);
				// 减去航向安装角误差
				m_GLSInfo.MeasInno_ELC.at(m_GLSInfo.nMeas_ELC + 0, 0, M1[0] - sin(m_pSINSOpt->m_RotAngleNHC[0] *D2R)*pELC->DMI_DATA.DMIVel);
				// 减去俯仰安装角误差
				m_GLSInfo.MeasInno_ELC.at(m_GLSInfo.nMeas_ELC + 1, 0, M1[2] - sin(m_pSINSOpt->m_RotAngleNHC[1] *D2R)*pELC->DMI_DATA.DMIVel);

				if (!bMeasDMI)
				{
					m_GLSInfo.MeasR_ELC.at(m_GLSInfo.nMeas_ELC + 0, m_GLSInfo.nMeas_ELC, SQR(0.3));
					m_GLSInfo.MeasR_ELC.at(m_GLSInfo.nMeas_ELC + 1, m_GLSInfo.nMeas_ELC + 1, SQR(0.3));
				}
				else
				{
					m_GLSInfo.MeasR_ELC.at(m_GLSInfo.nMeas_ELC + 0, m_GLSInfo.nMeas_ELC, SQR(0.15));
					m_GLSInfo.MeasR_ELC.at(m_GLSInfo.nMeas_ELC + 1, m_GLSInfo.nMeas_ELC + 1, SQR(0.10));

					if (m_pSINSOpt->m_RotAngleNHC[1] > 1.0) m_GLSInfo.MeasR_ELC.at(m_GLSInfo.nMeas_ELC + 1, m_GLSInfo.nMeas_ELC + 1, SQR(0.1 + sin(0.1 * m_pSINSOpt->m_RotAngleNHC[1] * D2R)*pELC->DMI_DATA.DMIVel));

					if (m_pSINSOpt->m_RotAngleNHC[0] > 2.0) m_GLSInfo.MeasR_ELC.at(m_GLSInfo.nMeas_ELC + 0, m_GLSInfo.nMeas_ELC + 0, SQR(0.2 + sin(0.5 * m_pSINSOpt->m_RotAngleNHC[0] * D2R)*pELC->DMI_DATA.DMIVel));
					if (m_pSINSOpt->m_RotAngleNHC[1] > 2.0) m_GLSInfo.MeasR_ELC.at(m_GLSInfo.nMeas_ELC + 1, m_GLSInfo.nMeas_ELC + 1, SQR(0.2 + sin(0.5 * m_pSINSOpt->m_RotAngleNHC[1] * D2R)*pELC->DMI_DATA.DMIVel));
				}

				m_GLSInfo.nMeas_ELC += NMeas_NHC;
			}
		}


		if(bMeasHGH)
		{
			double Hinv[9] = { 0.0 };
			m_SimpleEGM.setXYZ(m_GLSInfo.Position);
			m_SimpleEGM.getHeightConstraint(Hinv);

			m_GLSInfo.MeasH_ELC.at(m_GLSInfo.nMeas_ELC + 0, 0, Hinv[6]);
			m_GLSInfo.MeasH_ELC.at(m_GLSInfo.nMeas_ELC + 0, 1, Hinv[7]);
			m_GLSInfo.MeasH_ELC.at(m_GLSInfo.nMeas_ELC + 0, 2, Hinv[8]);
		
			double LLH[3];
			XYZ2LLH(m_GLSInfo.Position,LLH);
			
			m_GLSInfo.MeasInno_ELC.at(m_GLSInfo.nMeas_ELC, 0, (LLH[2] - pELC->NHC_DATA.NhcHgh));
			m_GLSInfo.MeasR_ELC.at(m_GLSInfo.nMeas_ELC, m_GLSInfo.nMeas_ELC, pELC->NHC_DATA.NhcHghP);
			m_GLSInfo.nMeas_ELC += NMeas_HGH;
		}


		if(bMeasHED)
		{
			double dAtt = 0.0;
			double att[3] = { 0.0 };
			double LLH[3];
			LLH2XYZ(m_GLSInfo.Position,LLH);

			// 外部输入的航向角为北向起算,将其转换为程序定义的角度
			// 详见CPose3D::Azimuth2Attitude()函数
			double yaw = 0.0;
			if (pELC->HED_DATA.Heading <= PI) yaw = -pELC->HED_DATA.Heading;
			else                     yaw = PI_2 - pELC->HED_DATA.Heading;

			// 机械编排的航向角
			Rbe2Attitude(m_GLSInfo.Rbn,LLH,att);

			// (att[0][0]-yaw)有可能在定义域两边,例如179和-179是同个角度,直接做差为358度,显然不对
			if(fabs(att[0] - yaw) > PI)
			{
				if(att[0] > 0.0) yaw += PI_2;
				else             yaw -= PI_2;
			}

			dAtt = yaw - att[0]; // 真值减去观测值

			// 如果差值太大,说明有问题,不做更新
			if(fabs(dAtt) < PI_12)
			{
				double* MisAngle = m_GLSInfo.StateX.x + m_GLSInfo.IMisAngle;

				double Rel[9];
				Rxyz2enu(LLH,Rel);

				double f[3] = { 0 };
				double ND = 0.0;
				double n[4] = { 0.0 };
				double d[4] = { 0.0 };
				double N = 0.0, D = 0.0;

				n[0] = Rel[2] * m_GLSInfo.Rbn[4] - Rel[1] * m_GLSInfo.Rbn[7];
				n[1] = Rel[0] * m_GLSInfo.Rbn[7] - Rel[2] * m_GLSInfo.Rbn[1];
				n[2] = Rel[1] * m_GLSInfo.Rbn[1] - Rel[0] * m_GLSInfo.Rbn[4];
				n[3] = Rel[0] * m_GLSInfo.Rbn[1] + Rel[1] * m_GLSInfo.Rbn[4] + Rel[0] * m_GLSInfo.Rbn[7];

				d[0] = Rel[5] * m_GLSInfo.Rbn[4] - Rel[4] * m_GLSInfo.Rbn[7];
				d[1] = Rel[3] * m_GLSInfo.Rbn[7] - Rel[5] * m_GLSInfo.Rbn[1];
				d[2] = Rel[4] * m_GLSInfo.Rbn[1] - Rel[3] * m_GLSInfo.Rbn[4];
				d[3] = Rel[3] * m_GLSInfo.Rbn[1] + Rel[4] * m_GLSInfo.Rbn[4] + Rel[5] * m_GLSInfo.Rbn[7];

				N = n[0] * MisAngle[0] + n[1] * MisAngle[1] + n[2] * MisAngle[2] + n[3];
				D = d[0] * MisAngle[0] + d[1] * MisAngle[1] + d[2] * MisAngle[2] + d[3];

				ND = N*N + D*D;

				f[0] = (N*d[0] - D*n[0]) / ND;
				f[1] = (N*d[1] - D*n[1]) / ND;
				f[2] = (N*d[2] - D*n[2]) / ND;

				m_GLSInfo.MeasH_ELC.at(m_GLSInfo.nMeas_ELC, m_GLSInfo.IMisAngle + 0, f[0]);
				m_GLSInfo.MeasH_ELC.at(m_GLSInfo.nMeas_ELC, m_GLSInfo.IMisAngle + 1, f[1]);
				m_GLSInfo.MeasH_ELC.at(m_GLSInfo.nMeas_ELC, m_GLSInfo.IMisAngle + 2, f[2]);
				m_GLSInfo.MeasInno_ELC.at(m_GLSInfo.nMeas_ELC, 0, dAtt);
				m_GLSInfo.MeasR_ELC.at(m_GLSInfo.nMeas_ELC, m_GLSInfo.nMeas_ELC, pELC->HED_DATA.HeadingP);
				
				m_GLSInfo.nMeas_ELC += NMeas_HED;
			}
			else
			{
				DumpException(string("The difference of heading angle updating is too much!"), LOGT_INFO, DUMP_LOCATION);
			}
		}


		if (bMeasZIH)
		{
			double LLH[3] = { 0.0 };
			double att[3] = { 0.0 };
			double attitude[3] = { 0.0 };
			double dHeading = 0.0;
			double dT = m_UpdateDeltT;

			XYZ2LLH(m_GLSInfo.Position, LLH);
			Rbe2Attitude(m_GLSInfo.Rbn, LLH, att);

			if (m_pSINSOpt->m_bAzimuth)
				Azimuth2Attitude(m_pSINSSol->m_Att, attitude);
			else
				M31EQU(m_pSINSSol->m_Att, attitude);

			dHeading = att[0] - attitude[0];

			m_GLSInfo.MeasH_ELC.at(m_GLSInfo.nMeas_ELC, m_GLSInfo.IGyrosBias + 0, (-sin(attitude[1]) / cos(attitude[2]) *dT));
			m_GLSInfo.MeasH_ELC.at(m_GLSInfo.nMeas_ELC, m_GLSInfo.IGyrosBias + 1, 0.0);
			m_GLSInfo.MeasH_ELC.at(m_GLSInfo.nMeas_ELC, m_GLSInfo.IGyrosBias + 2, (-cos(attitude[1]) / cos(attitude[2]) *dT));
			m_GLSInfo.MeasInno_ELC.at(m_GLSInfo.nMeas_ELC, 0, dHeading);
			m_GLSInfo.MeasR_ELC.at(m_GLSInfo.nMeas_ELC, m_GLSInfo.nMeas_ELC, m_pSINSSol->m_AttP[0]);//待定

			m_GLSInfo.nMeas_ELC += NMeas_ZIH;
		}


		if (bMeasCUP)
		{
			m_GLSInfo.MeasH_ELC.at(m_GLSInfo.nMeas_ELC + 0, 0, 1.0);
			m_GLSInfo.MeasH_ELC.at(m_GLSInfo.nMeas_ELC + 1, 1, 1.0);
			m_GLSInfo.MeasH_ELC.at(m_GLSInfo.nMeas_ELC + 2, 2, 1.0);
			double nlarm_CUP[3];
			M33XM31(m_GLSInfo.Rbn, pELC->CUP_DATA.LeverArm, nlarm_CUP);
			double nlarmX_CUP[9];
			MatrixSkewSymmetric(nlarm_CUP, nlarmX_CUP);
			m_GLSInfo.MeasH_ELC.setSub(m_GLSInfo.nMeas_ELC, m_GLSInfo.IMisAngle, 3, 3, nlarmX_CUP);

			M31M31(m_GLSInfo.Position, nlarm_CUP, M1);
			M31_M31(M1, pELC->CUP_DATA.CUPTPos, M2);
			m_GLSInfo.MeasInno_ELC.setSub(m_GLSInfo.nMeas_ELC, 0, 3, 1, M2);

			// 将L系下的CUP方差转到e系下
			double LLH[3] = { 0.0 };
			XYZ2LLH(pELC->CUP_DATA.CUPTPos, LLH);
			double J[9] = { 0.0 };
			double W = sqrt(1 - gs_WGS84_e2 * SQR(sin(LLH[0])));
			double rmh = gs_WGS84_a * (1 - gs_WGS84_e2) / pow(W, 3) + LLH[2];
			double rnh = gs_WGS84_a / W + LLH[2];
			CGeodesy::Geodetic_Jacobian(LLH[0], LLH[1], rmh, rnh, J, false, true);
			double CUPTPosP[9] = { 0.0 };
			MatrixMultiply_HPHT(3, 3, J, pELC->CUP_DATA.CUPTPosP, false, CUPTPosP);

			m_GLSInfo.MeasR_ELC.setSub(m_GLSInfo.nMeas_ELC, m_GLSInfo.nMeas_ELC, 3, 3, CUPTPosP);

			m_GLSInfo.nMeas_ELC += NMeas_CUP;
		}


		if(m_GLSInfo.nMeas_ELC != 0)
		{
			if (m_GLS.initGLS(m_GLSInfo.StateX.row, m_GLSInfo.nMeas_ELC, false) == false) return false;
			m_GLS.m_H.setSub(0, 0, m_GLSInfo.nMeas_ELC, m_GLSInfo.StateX.row, m_GLSInfo.MeasH_ELC);
			m_GLS.m_Inno.setSub(0, 0, m_GLSInfo.nMeas_ELC, 1, m_GLSInfo.MeasInno_ELC);
			m_GLS.m_R.setSub(0, 0, m_GLSInfo.nMeas_ELC, m_GLSInfo.nMeas_ELC, m_GLSInfo.MeasR_ELC);
			return true;
		}

		return false;
	}


	








	// ************************************************************************************************************************************** //
	//  previous class : CLooseCouplingECEF ;     next class : CLooseCouplingLocal                                                            //
	// ************************************************************************************************************************************** //








	CLooseCouplingLocal::CLooseCouplingLocal()
	{
		
	}

	CLooseCouplingLocal::~CLooseCouplingLocal()
	{
		
	}

	bool CLooseCouplingLocal::initLC(bool bReverse)
	{
		m_bReverse = bReverse;
		m_pSINSOpt->m_NaviSys = NAVI_L;

		if(initLCBase())
		{
			setSINSKF_EarthPara(m_pSINSOpt, &m_GLSInfo, &m_SimpleEGM, m_GLSInfo.Position,m_GLSInfo.Velocity);

			// 外部提供的噪声参数是ENU方向的,而L系下的状态为大地坐标系(经纬度),需要下面的转换
			// 当地坐标系ENU转换为大地坐标系
			m_GLSInfo.StateP.at(m_GLSInfo.IPos + 0, m_GLSInfo.IPos + 0, SQR(m_pSINSOpt->m_SSP.ISDV_DeltPosY) / SQR(m_GLSInfo.rmh));
			m_GLSInfo.StateP.at(m_GLSInfo.IPos + 1, m_GLSInfo.IPos + 1, SQR(m_pSINSOpt->m_SSP.ISDV_DeltPosX) / SQR(m_GLSInfo.rnh*cos(m_GLSInfo.Position[0])));									
			return true;
		}
		
		return false;
	}


	
	bool CLooseCouplingLocal::setMEQ(const int iter, const int ECLType)
	{
		bool bMeasPOS = m_GLSInfo.bMeasPOS && (ECLType&ELCTYPE_POS) && MatrixTrace(3, 3, m_pELCFFStream->m_ELCData.GNSS_DATA.XYZPosP) > 0.0;
		bool bMeasVEL = m_GLSInfo.bMeasVEL && (ECLType&ELCTYPE_VEL) && MatrixTrace(3, 3, m_pELCFFStream->m_ELCData.GNSS_DATA.XYZVelP) > 0.0;
		bool bMeasATT = m_GLSInfo.bMeasATT && (ECLType&ELCTYPE_ATT) && MatrixTrace(3, 3, m_pELCFFStream->m_ELCData.GNSS_DATA.AttP) > 0.0;
		bool bMeasHED = m_GLSInfo.bMeasHED && (ECLType&ELCTYPE_HED) && m_pELCFFStream->m_ELCData.HED_DATA.HeadingP > 0.0;
		bool bMeasDMI = m_GLSInfo.bMeasDMI && (ECLType&ELCTYPE_DMI) && m_pELCFFStream->m_ELCData.DMI_DATA.DMIVelP > 0.0;
		bool bMeasZUP = m_GLSInfo.bMeasZUP && (ECLType&ELCTYPE_ZUP);
		bool bMeasZIH = m_GLSInfo.bMeasZIH && (ECLType&ELCTYPE_ZIH);
		bool bMeasCUP = m_GLSInfo.bMeasCUP && (ECLType&ELCTYPE_CUP) && MatrixTrace(3, 3, m_pELCFFStream->m_ELCData.CUP_DATA.CUPTPosP) > 0.0;
		bool bMeasNHC = m_GLSInfo.bMeasNHC && (ECLType&ELCTYPE_NHC);
		bool bMeasHGH = m_GLSInfo.bMeasHGH && (ECLType&ELCTYPE_HGH);
		bMeasHGH = false; // 高程和垂向速度约束应该等价,作用不大

		// 满足以下条件,说明位置解算的不好,直接不用位置
		if (bMeasPOS)
		{
			//	if (m_pELCFFStream->m_ELCData.DDOP >= 50.0) bMeasurePos = false;
			//	if (m_pELCFFStream->m_ELCData.Q > 7) bMeasurePos = false;
			//	if (m_pELCFFStream->m_ELCData.AmbFix < 0) bMeasurePos = false;
		}

		// 所有观测更新均为false,则直接返回
		if ((!bMeasPOS) && (!bMeasVEL) && (!bMeasATT) && (!bMeasHED) && (!bMeasDMI) &&
			(!bMeasZUP) && (!bMeasZIH) && (!bMeasCUP) && (!bMeasNHC) && (!bMeasHGH))
			return false;

		ELC_DATA* pELC = &m_pELCFFStream->m_ELCData;

		double MeasPos[3];
		double MeasVel[3];
		double MeasPosP[9];
		double MeasVelP[9];
		double rE2L[9];

		// 得到L系下的坐标
		XYZ2LLH(pELC->GNSS_DATA.XYZPos, MeasPos);
		// 得到L系下的速度
		Rxyz2enu(MeasPos, rE2L);
		M33XM31(rE2L, pELC->GNSS_DATA.XYZVel, MeasVel);
		
		// 转换位置,速度协方差矩阵
		double J[9];
		setSINSKF_EarthPara(m_pSINSOpt, &m_GLSInfo, &m_SimpleEGM, MeasPos, MeasVel, true, false);
		CGeodesy::Geodetic_Jacobian(MeasPos[0], MeasPos[1], m_GLSInfo.rmh, m_GLSInfo.rnh, J, true, m_GLSInfo.bTopoCoor);
		MatrixMultiply_HPHT(3, 3, J, pELC->GNSS_DATA.XYZPosP, false, MeasPosP);

		MatrixMultiply_HPHT(3, 3, J, pELC->GNSS_DATA.XYZVelP, false, MeasVelP);
		
		// 1. 位置、速度、姿态修正,得到较为准确的值,减弱观测方程非线性
		M31_M31(m_SINSMech.m_Position, m_GLSInfo.StateX.x + m_GLSInfo.IPos, m_GLSInfo.Position);
		M31_M31(m_SINSMech.m_Velocity, m_GLSInfo.StateX.x + m_GLSInfo.IVel, m_GLSInfo.Velocity);
		CorrectMisAngle(m_SINSMech.m_Rbn, m_GLSInfo.StateX.x + m_GLSInfo.IMisAngle, m_GLSInfo.Rbn);
		UnitOrthoRotMatrix(m_GLSInfo.Rbn);

		// 2. m_MeasureHMatrix 等清零
		m_GLSInfo.MeasH_ELC.zero();
		m_GLSInfo.MeasR_ELC.zero();
		m_GLSInfo.MeasInno_ELC.zero();
		m_GLSInfo.nMeas_ELC = 0;

		double M1[3], M2[3], R1[9];

		double nlarm[3];
		M33XM31(m_GLSInfo.Rbn, m_pSINSOpt->m_LeverArmbGNSS, nlarm);
		double nlarmX[9];
		MatrixSkewSymmetric(nlarm, nlarmX);
		
		if(bMeasPOS)
		{
			double cosp = cos(m_GLSInfo.Position[0]);
			double dps = m_GLSInfo.rmh;
			double dls = m_GLSInfo.rnh * cosp;
			double MeasurePos[3];

			// 是否将经纬度转换成坐标扰动，避免数值问题，有误？？？
			if(m_GLSInfo.bTopoCoor)
			{
				m_GLSInfo.MeasH_ELC.at(0, 1, dls);
				m_GLSInfo.MeasH_ELC.at(1, 0, dps);
				m_GLSInfo.MeasH_ELC.at(2, 2, 1.0);
				m_GLSInfo.MeasH_ELC.setSub(0, m_GLSInfo.IMisAngle, 3, 3, nlarmX);
				

				M31_M31(m_GLSInfo.Position, MeasPos, MeasurePos);
				MeasurePos[0] *= dps;
				MeasurePos[1] *= dls;

				double tmp = MeasurePos[0];
				MeasurePos[0] = MeasurePos[1];
				MeasurePos[1] = tmp;
				M31M31(nlarm, MeasurePos);
			}
			else
			{
				m_GLSInfo.MeasH_ELC.at(0, 0, 1.0);
				m_GLSInfo.MeasH_ELC.at(1, 1, 1.0);
				m_GLSInfo.MeasH_ELC.at(2, 2, 1.0);
				
				// 通过将式子展开可以推导得到
				double tlarm[3];
				tlarm[0] = nlarm[1]/dps; 
				tlarm[1] = nlarm[0]/dls; 
				tlarm[2] = nlarm[2];

				double tlarmX[9];
				tlarmX[0] = nlarm[2]/dps; tlarmX[2] =-nlarm[0]/dps;
				tlarmX[4] =-nlarm[2]/dls; tlarmX[5] = nlarm[1]/dls;
				tlarmX[6] =-nlarm[1];     tlarmX[7] = nlarm[0];

				m_GLSInfo.MeasH_ELC.setSub(0, m_GLSInfo.IMisAngle, 3, 3, tlarmX);
				M31M31(m_GLSInfo.Position, tlarm, M1);
				M31_M31(M1, MeasPos, MeasurePos);
			}

			m_GLSInfo.MeasInno_ELC.setSub(0, 0, 3, 1, MeasurePos);
			m_GLSInfo.MeasR_ELC.setSub(0, 0, 3, 3, MeasPosP);
			
			m_GLSInfo.nMeas_ELC += NMeas_POS;
		}


		if(bMeasVEL)
		{
			double blarmX[9];
			MatrixSkewSymmetric(m_pSINSOpt->m_LeverArmbGNSS, blarmX);

			double win[3];
			M31M31(m_GLSInfo.Wnie, m_GLSInfo.Wnen, win);

			// dMatrix larmw  = Cross3(nlarm,m_pIMUFFStream->m_bGyros) + Cross3(win,nlarm); // 容易犯错误,TC中影响很明显
			double larmw[3];
			CrossM3(win, nlarm, M1);
			CrossM3(m_pSINSOpt->m_LeverArmbGNSS, m_pIMUFFStream->m_bGyros, M2);
			M33XM31(m_GLSInfo.Rbn, M2, larmw);
			M31M31(M1, larmw);

			double larmwX[9];
			MatrixSkewSymmetric(larmw, larmwX);

			m_GLSInfo.MeasH_ELC.at(m_GLSInfo.nMeas_ELC + 0, 3, 1.0);
			m_GLSInfo.MeasH_ELC.at(m_GLSInfo.nMeas_ELC + 1, 4, 1.0);
			m_GLSInfo.MeasH_ELC.at(m_GLSInfo.nMeas_ELC + 2, 5, 1.0);
			m_GLSInfo.MeasH_ELC.setSub(m_GLSInfo.nMeas_ELC, m_GLSInfo.IMisAngle, 3, 3, larmwX, true);

			M33XM33(m_GLSInfo.Rbn, blarmX, R1);
			m_GLSInfo.MeasH_ELC.setSub(m_GLSInfo.nMeas_ELC, m_GLSInfo.IGyrosBias, 3, 3, R1, true);

			if (m_pSINSOpt->m_bAddGyrosBiasTob)
			{
				m_GLSInfo.MeasH_ELC.setSub(m_GLSInfo.nMeas_ELC, m_GLSInfo.IGyrosBiasTob, 3, 3, R1, true);
			}

			if (m_pSINSOpt->m_bAddGyrosBiasScale)
			{
				M33XM33(m_GLSInfo.Rbn, blarmX, R1);
				R1[0] *= m_pIMUFFStream->m_bGyros[0]; R1[3] *= m_pIMUFFStream->m_bGyros[0]; R1[6] *= m_pIMUFFStream->m_bGyros[0];
				R1[1] *= m_pIMUFFStream->m_bGyros[1]; R1[4] *= m_pIMUFFStream->m_bGyros[1]; R1[7] *= m_pIMUFFStream->m_bGyros[1];
				R1[2] *= m_pIMUFFStream->m_bGyros[2]; R1[5] *= m_pIMUFFStream->m_bGyros[2]; R1[8] *= m_pIMUFFStream->m_bGyros[2];
				m_GLSInfo.MeasH_ELC.setSub(m_GLSInfo.nMeas_ELC, m_GLSInfo.IGyrosBiasScale, 3, 3, R1, true);
			}
			
			if (m_pSINSOpt->m_bAddGyrosBiasOrthog)
			{
				M33XM33(m_GLSInfo.Rbn, blarmX, R1);

				double RbeM[18];
				double M[18] = { 0.0 };
				M[0 * 6 + 0] = m_pIMUFFStream->m_bGyros[1]; M[0 * 6 + 1] = m_pIMUFFStream->m_bGyros[2];
				M[1 * 6 + 2] = m_pIMUFFStream->m_bGyros[0]; M[1 * 6 + 3] = m_pIMUFFStream->m_bGyros[2];
				M[2 * 6 + 4] = m_pIMUFFStream->m_bGyros[0]; M[2 * 6 + 5] = m_pIMUFFStream->m_bGyros[1];
				MatrixMultiply(3, 3, R1, 3, 6, M, RbeM);

				m_GLSInfo.MeasH_ELC.setSub(m_GLSInfo.nMeas_ELC, m_GLSInfo.IGyrosBiasOrthog, 3, 6, RbeM, true);
			}

			M31_M31(m_GLSInfo.Velocity, larmw, M1);
			M31_M31(M1, pELC->GNSS_DATA.XYZVel, M2);
			m_GLSInfo.MeasInno_ELC.setSub(m_GLSInfo.nMeas_ELC, 0, 3, 1, M2, false);
			m_GLSInfo.MeasR_ELC.setSub(m_GLSInfo.nMeas_ELC, m_GLSInfo.nMeas_ELC, 3, 3, pELC->GNSS_DATA.XYZVelP);
			m_GLSInfo.nMeas_ELC += NMeas_VEL;
		}


		if(bMeasDMI)
		{
			
		}


		if(bMeasNHC)
		{
			if(_FRAC((m_pELCFFStream->m_time - m_UpdateTime)/m_pELCFFStream->m_DeltT) == 0.0)
			{
				m_GLSInfo.MeasH_ELC.at(m_GLSInfo.nMeas_ELC + 0, 3, m_GLSInfo.Rbn[0]);
				m_GLSInfo.MeasH_ELC.at(m_GLSInfo.nMeas_ELC + 0, 4, m_GLSInfo.Rbn[3]);
				m_GLSInfo.MeasH_ELC.at(m_GLSInfo.nMeas_ELC + 0, 5, m_GLSInfo.Rbn[6]);
				m_GLSInfo.MeasH_ELC.at(m_GLSInfo.nMeas_ELC + 0, 6, m_GLSInfo.Rbn[6] * m_GLSInfo.Velocity[1] - m_GLSInfo.Rbn[3] * m_GLSInfo.Velocity[2]);
				m_GLSInfo.MeasH_ELC.at(m_GLSInfo.nMeas_ELC + 0, 7, m_GLSInfo.Rbn[0] * m_GLSInfo.Velocity[2] - m_GLSInfo.Rbn[6] * m_GLSInfo.Velocity[0]);
				m_GLSInfo.MeasH_ELC.at(m_GLSInfo.nMeas_ELC + 0, 8, m_GLSInfo.Rbn[3] * m_GLSInfo.Velocity[0] - m_GLSInfo.Rbn[0] * m_GLSInfo.Velocity[1]);

				m_GLSInfo.MeasH_ELC.at(m_GLSInfo.nMeas_ELC + 1, 3, m_GLSInfo.Rbn[2]);
				m_GLSInfo.MeasH_ELC.at(m_GLSInfo.nMeas_ELC + 1, 4, m_GLSInfo.Rbn[5]);
				m_GLSInfo.MeasH_ELC.at(m_GLSInfo.nMeas_ELC + 1, 5, m_GLSInfo.Rbn[8]);
				m_GLSInfo.MeasH_ELC.at(m_GLSInfo.nMeas_ELC + 1, 6, m_GLSInfo.Rbn[8] * m_GLSInfo.Velocity[1] - m_GLSInfo.Rbn[5] * m_GLSInfo.Velocity[2]);
				m_GLSInfo.MeasH_ELC.at(m_GLSInfo.nMeas_ELC + 1, 7, m_GLSInfo.Rbn[2] * m_GLSInfo.Velocity[2] - m_GLSInfo.Rbn[8] * m_GLSInfo.Velocity[0]);
				m_GLSInfo.MeasH_ELC.at(m_GLSInfo.nMeas_ELC + 1, 8, m_GLSInfo.Rbn[5] * m_GLSInfo.Velocity[0] - m_GLSInfo.Rbn[2] * m_GLSInfo.Velocity[1]);

				MatrixTranspose(3, 3, m_GLSInfo.Rbn, R1);
				M33XM31(R1, m_GLSInfo.Velocity, M1);

				// 取出XZ方向的速度作为观测		
				m_GLSInfo.MeasInno_ELC.at(m_GLSInfo.nMeas_ELC, 0, M1[0]);
				m_GLSInfo.MeasInno_ELC.at(m_GLSInfo.nMeas_ELC + 1, 0, M1[2]);

				m_GLSInfo.MeasR_ELC.at(m_GLSInfo.nMeas_ELC, m_GLSInfo.nMeas_ELC, SQR(0.1));
				m_GLSInfo.MeasR_ELC.at(m_GLSInfo.nMeas_ELC + 1, m_GLSInfo.nMeas_ELC + 1, SQR(0.1));

				m_GLSInfo.nMeas_ELC += NMeas_NHC;
			}
		}


		if (bMeasHGH)
		{
			if(_FRAC((m_pELCFFStream->m_time - m_UpdateTime)/m_pELCFFStream->m_DeltT) == 0.0)
			{
				m_GLSInfo.MeasH_ELC.at(m_GLSInfo.nMeas_ELC, 2, 1.0);
				m_GLSInfo.MeasInno_ELC.at(m_GLSInfo.nMeas_ELC, 0, (m_GLSInfo.Position[2] - pELC->NHC_DATA.NhcHgh));
				m_GLSInfo.MeasR_ELC.at(m_GLSInfo.nMeas_ELC, m_GLSInfo.nMeas_ELC, pELC->NHC_DATA.NhcHghP);
				m_GLSInfo.nMeas_ELC += NMeas_HGH;
			}
		}


		if(bMeasHED)
		{
			
		}


		if(m_GLSInfo.nMeas_ELC != 0)
		{
			if (m_GLS.initGLS(m_GLSInfo.StateX.row, m_GLSInfo.nMeas_ELC, false) == false) return false;
			m_GLS.m_H.setSub(0, 0, m_GLSInfo.nMeas_ELC, m_GLSInfo.StateX.row, m_GLSInfo.MeasH_ELC);
			m_GLS.m_Inno.setSub(0, 0, m_GLSInfo.nMeas_ELC, 1, m_GLSInfo.MeasInno_ELC);
			m_GLS.m_R.setSub(0, 0, m_GLSInfo.nMeas_ELC, m_GLSInfo.nMeas_ELC, m_GLSInfo.MeasR_ELC);
			return true;
		}

		return false;
	}








	// ************************************************************************************************************************************** //
	//  previous class: CLooseCouplingLocal ;     next class :                                                                                //
	// ************************************************************************************************************************************** //









} // end namespace

