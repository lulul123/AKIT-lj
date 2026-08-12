/**
* @file
* @brief    General Least Square Estimator or Kalman Filter
* @details  广义最小二乘估计器,KF滤波器 \n
* @author   Feng Zhu. Email: fzhu@whu.edu.cn
* @date     2018/01/17
* @version  2.3.20230417
* @par      Copyright(c) 2012-2021 School of Geodesy and Geomatics, University of Wuhan. All Rights Reserved.
*           POSMind Software Kernel, named IPS (Inertial Plus Sensing), this file is a part of IPS.
* @par      History:
*           2018/01/17,Feng Zhu, new \n
*/


#ifndef BASETK_LSESTIMATOR_H
#define BASETK_LSESTIMATOR_H

#include "BASEDLL/BaseSDC.h"
#include "BASEDLL/BaseMatrix.h"

namespace basetk
{
	class IPS_DLLEXPORT CLSEstimator
	{

	public:

		CLSEstimator();
		virtual ~CLSEstimator();

		bool initLSQ(int nState, int nMeas, bool bPDiagonal);
		
		///< Least-Square Estimate X=(BTPB)^-1*BTPL
		bool LSQ(bool bPostEst = false);
		
		///< 输入
		Mat m_B;				///< 观测矩阵
		Mat m_P;				///< 观测权阵,不是协防差阵
		Mat m_L;				///< 观测值
		
		///< 输出
		Mat m_StateX;			///< 估计的状态
		Mat m_StateP;			///< 估计的状态协方差
		Mat m_V;				///< 改正数 V = BX - W
		
		double m_Sigma0;		///< 单位权中误差(原始的中误差,提供给外部判断解的质量)
		double m_Sigma;			///< 单位权中误差(处理过后的,与协因数矩阵相乘后,得到状态协方差阵)
		double m_RMS;			///< 验后残差RMS

		int m_iCons;			///< 状态约束起点
		Mat m_ConsStateX;		///< 状态约束值
		Mat m_ConsStateP;		///< 状态约束斜方差

	private:

		bool m_bInit;			///< 是否初始化LSQ
		bool m_bPDiagonal;		///< 观测权阵P是否为对角线
		int  m_nState;			///< 状态数
		int  m_nMeas;			///< 观测数
		Mat  m_W;				///< BTPL
		double m_SigmaPre;		///< 当前sigma不能用时,用前一个代入

		void FreeLSQ();
		void ClearLSQ();
	};


	class IPS_DLLEXPORT CGLSEstimator
	{

	public:

		CGLSEstimator();
		
		virtual ~CGLSEstimator();

		// lj:修改了该函数的参数 加入更新时间
		void KFPrediction(double dt, const Mat CF, const Mat Qc, const Mat StateP, Mat Phi, Mat StatePPre);
		//lj修改 2026-08-11：成功的GNSS位置更新结束当前累计区间
		void FinishGNSSInterval(Mat PPred);

		// lj: 加入一个时间
		int updateTime;

		bool initGLS(int nState, int nMeas, bool bPDiagonal);
		bool GLS();
		
		void setXP(vector<int>& index, Mat X, Mat P);
		void getXP(vector<int>& index, Mat X, Mat P);

		bool m_bMDB;			///< 是否需要MDB
		bool m_bInnoChisq;		///< 是否需要新息卡方分布
		bool m_bResiChisq;		///< 是否需要残差卡方分布

		///< 输入
		Mat m_StateXp;			///< 估计的状态, nS x 1
		Mat m_StatePp;			///< 估计的状态协方差, nS x nS
		Mat m_H;				///< 观测矩阵, nM x nS
		Mat m_R;				///< 观测协防差阵, nM x nM
		Mat m_Inno;				///< 新息, nM x 1

		///< 输出
		Mat m_StateX;			///< 估计的状态, nS x 1
		Mat m_StateP;			///< 估计的状态协方差, nS x nS
		double m_dXl;			///< 状态变化量的模,作为迭代收敛的判断

		Mat m_Resi;				///< 残差, nM x 1, 当bResiChisq=true时有输出
		Mat m_Qi;				///< 新息协方差, nM x nM, 当bInnoChisq=true时有输出
		Mat m_Qv;				///< 残差协方差, nM x nM, 当bResiChisq=true时有输出
		Mat m_Ry;				///< 观测值的多余分量观测, nM x nM

		double m_InnoChisq;		///< 新息卡方分布值
		double m_ResiChisq;		///< 残差卡方分布值
		double m_InnoSigma0;	///< 新息sigma0
		double m_ResiSigma0;	///< 残差sigma0
		double m_RMS;			///< 验后残差RMS

		bool m_bMeasVarPre;		///< 预报的观测方差是否大于本身的预报方差

	private:

		bool m_bInit;			///< 是否初始化LSQ

		int m_nState;			///< 状态数
		int m_nMeas;			///< 观测数

		///< 中间变量
		Mat m_Rp;				///< 预报观测值的协防差阵, nM x nM
		Mat m_K;				///< Kalman滤波增益矩阵, nS x nM
		Mat m_KFTmp_F;
		Mat m_KFTmp_Q;
		//lj修改 2026-08-11：GNSS区间累计状态改由cpp内部实例映射保存，不改变导出类布局
		bool m_bPDiagonal;		///< 观测权阵P是否为对角线

		void FreeGLS(bool bFreeState = true);
		void ClearGLS();

	private:

	};



	class IPS_DLLEXPORT CGLSEstimatorX
	{

	public:

		CGLSEstimatorX();

		virtual ~CGLSEstimatorX();

		void KFPrediction(double dt, const Mat CF, const Mat Qc, const Mat StateP, Mat Phi, Mat StatePPre);

		bool initGLS(int nState, int nMeas, bool bPDiagonal);
		bool GLS();

		void setXP(vector<int>& index, Mat X, Mat P);
		void getXP(vector<int>& index, Mat X, Mat P);

		bool m_bMDB;			///< 是否需要MDB
		bool m_bInnoChisq;		///< 是否需要新息卡方分布
		bool m_bResiChisq;		///< 是否需要残差卡方分布

		///< 输入
		Mat m_StateXp;			///< 估计的状态, nS x 1
		Mat m_StatePp;			///< 估计的状态协方差, nS x nS
		Mat m_H;				///< 观测矩阵, nM x nS
		Mat m_R;				///< 观测协防差阵, nM x nM
		Mat m_Inno;				///< 新息, nM x 1



		///< 输出
		Mat m_StateX;			///< 估计的状态, nS x 1
		Mat m_StateP;			///< 估计的状态协方差, nS x nS
		double m_dXl;			///< 状态变化量的模,作为迭代收敛的判断

		Mat m_Resi;				///< 残差, nM x 1, 当bResiChisq=true时有输出
		Mat m_Qi;				///< 新息协方差, nM x nM, 当bInnoChisq=true时有输出
		Mat m_Qv;				///< 残差协方差, nM x nM, 当bResiChisq=true时有输出
		Mat m_Ry;				///< 观测值的多余分量观测, nM x nM

		double m_InnoChisq;		///< 新息卡方分布值
		double m_ResiChisq;		///< 残差卡方分布值
		double m_InnoSigma0;	///< 新息sigma0
		double m_ResiSigma0;	///< 残差sigma0
		double m_RMS;			///< 验后残差RMS

		bool m_bMeasVarPre;		///< 预报的观测方差是否大于本身的预报方差

	private:

		bool m_bInit;			///< 是否初始化LSQ

		int m_nState;			///< 状态数
		int m_nMeas;			///< 观测数

		Mat m_HTmp;
		Mat m_InnoTmp;
		Mat m_RpTmp;

		///< 中间变量
		Mat m_KFTmp_F;
		Mat m_KFTmp_Q;

		bool m_bPDiagonal;		///< 观测权阵P是否为对角线

		double* m_SRKF_K;
		double* m_SRKF_a;
		double* m_Rp_S;
		double* m_R_S;

		///< 当观测数小于状态数的时候,使用EKF效率高
		bool GLS1_EKF();		///< 处理R对对角阵的
		bool GLS2_EKF();		///< 处理R为非对角阵的

		///< 当观测数大于状态数的时候,使用LSQ效率高
		bool GLS1_LSQ();		///< 处理R对对角阵的
		bool GLS2_LSQ();		///< 处理R为非对角阵的

		void FreeGLS(bool bFreeState = true);
		void ClearGLS();

	private:

	};
}


#endif
