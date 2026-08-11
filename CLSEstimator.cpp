#include "BASEDLL/CLSEstimator.h"
#include "BASEDLL/BaseMath.h"
#include "BASEDLL/BaseApp.h"
#include "BASEDLL/BaseTime.h"

namespace basetk
{


	CLSEstimator::CLSEstimator()
	{
		m_nState = 0;
		m_nMeas = 0;
		m_bPDiagonal = false;
		m_bInit = false;
		m_Sigma = 1.0;
		m_Sigma0 = 0.0;
		m_SigmaPre = 1.0;
		m_RMS = 0.0;
		m_iCons = -1;
	}


	CLSEstimator::~CLSEstimator()
	{
		FreeLSQ();
	}


	void CLSEstimator::FreeLSQ()
	{
		m_StateX.freeMatrix();
		m_StateP.freeMatrix();
		m_B.freeMatrix();
		m_P.freeMatrix();
		m_L.freeMatrix();
		m_W.freeMatrix();
		m_V.freeMatrix();
	}


	void CLSEstimator::ClearLSQ()
	{
		m_B.zero();
		m_L.zero();
		m_P.zero();
	}


	bool CLSEstimator::initLSQ(int nState, int nMeas, bool bPDiagonal)
	{
		if (nState == m_nState && nMeas == m_nMeas)
		{
			ClearLSQ();
		}
		else
		{
			FreeLSQ();

			m_StateX.mat(nState, 1);
			m_StateP.mat(nState, nState);
			m_B.mat(nMeas, nState);
			m_P.mat(nMeas, nMeas);
			m_L.mat(nMeas, 1);
			m_W.mat(nState, 1);
			m_V.mat(nMeas, 1);
		}

		m_nState = nState;
		m_nMeas = nMeas;
		m_bPDiagonal = bPDiagonal;
		m_bInit = true;

		return true;
	}


	bool CLSEstimator::LSQ(bool bPostEst)
	{
		if (m_bInit == false)
		{
			DumpException("Forget to init least squares.", LOGT_INFO | LOGT_AFX, DUMP_LOCATION);
			return false;
		}

		MatrixMultiply_BTPB(m_nMeas, m_nState, m_B.x, m_P.x, m_bPDiagonal, m_StateP.x);
		MatrixMultiply_BTPL(m_nMeas, m_nState, m_B.x, m_P.x, m_L.x, m_bPDiagonal, m_W.x);

		if (m_iCons >= 0 && m_ConsStateX.row > 0 && m_ConsStateP.row > 0)
		{
			const int row = m_ConsStateP.row;
			const int col = m_ConsStateP.col;
			
			Mat M1(row, col), M2(row, col), M3(row, col);
			M1 = m_ConsStateP;
			M1.inv_sp();
			m_StateP.getSub(m_iCons, m_iCons, row, col, M2);
			M3.add(M1, M2);
			m_StateP.setSub(m_iCons, m_iCons, row, col, M3);

			Mat M4(row, 1), M5(row, 1);
			m_W.getSub(m_iCons, 0, row, 1, M4);
			M5.multiply(M1, m_ConsStateX);
			M5.add2(M4);
			m_W.setSub(m_iCons, 0, row, 1, M5);

			m_ConsStateX.freeMatrix();
			m_ConsStateP.freeMatrix();

			M1.freeMatrix(); M2.freeMatrix(); M3.freeMatrix();
			M4.freeMatrix(); M5.freeMatrix();

			m_iCons = -1;
		}

		
		if (m_StateP.inv_sp() == false)
		{
			string msg("Fail to perform least squares, the co-factor matrix is singular!");
			DumpException(msg, LOGT_ERROR | LOGT_AFX, DUMP_LOCATION);
			return false;
		}

		m_StateX.multiply(m_StateP, m_W);

		m_V.multiply(m_B, m_StateX);
		m_V.subtract_r(m_L); // V = B*X - L

		if (bPostEst)
		{
			m_Sigma = 1.0;
			m_RMS = 0.0;

			if (m_B.row > m_B.col)
			{
				MatrixMultiply_BTPB(m_V.row, m_V.col, m_V.x, m_P.x, m_bPDiagonal, &m_Sigma);
				m_Sigma = m_Sigma / (m_B.row - m_B.col);
				m_Sigma0 = xsqrt(m_Sigma);
				if (m_Sigma < 1.0) m_Sigma = 1.0; // sigma不要小于1.0
				m_StateP.scale(m_Sigma);
				m_P.scale(1.0 / m_Sigma);

				m_RMS = MatrixDot(m_V.row, m_V.col, m_V.x, m_V.x);
				m_RMS = xsqrt(m_RMS / m_V.row);
			}
			else
			{
				m_StateP.scale(m_SigmaPre);
				m_P.scale(1.0 / m_SigmaPre);
			}
	
			if (m_B.row > m_B.col) m_SigmaPre = m_Sigma;
			m_Sigma = xsqrt(m_SigmaPre);
		}

		if(0)
		{
			FILE* fp = fopen("D:\\LS.txt","w");
			m_StateX.write(fp);
			m_StateP.write(fp);
			m_B.write(fp);
			m_L.write(fp);
			m_P.write(fp);
			fclose(fp);
		}
		
		return true;
	}








	// ************************************************************************************************************************************** //
	//  previous class: CPPPointBase ;     next class : CPPPointIFLC                                                                          //
	// ************************************************************************************************************************************** //








	
	CGLSEstimator::CGLSEstimator()
	{
		m_bInit = false;
		m_bMDB = true;
		m_bPDiagonal = false;
		m_bInnoChisq = true;
		m_bResiChisq = true;
		m_nState = 0;
		m_nMeas = 0;
		//lj修改 2026-08-11：初始化GNSS成功更新区间的累计状态
		updateTime = 0;
		m_AccInitialized = false;
		m_StartNewInterval = true;
		m_HasSuccessfulGNSSBoundary = false;
		m_AccStepCount = 0;
	}


	CGLSEstimator::~CGLSEstimator()
	{
		FreeGLS(true);
	}


	bool CGLSEstimator::initGLS(int nState, int nMeas, bool bPDiagonal)
	{
		if (nState == 0 || nMeas == 0) return false;

		if (nMeas != m_nMeas || nState != m_nState)
		{
			FreeGLS(nState != m_nState);
			m_H.mat(nMeas, nState);
			m_R.mat(nMeas, nMeas);
			m_Inno.mat(nMeas, 1);
			m_Rp.mat(nMeas, nMeas);
			m_K.mat(nState, nMeas);
			if (m_bResiChisq)           m_Resi.mat(nMeas, 1);
			if (m_bInnoChisq)           m_Qi.mat(nMeas, nMeas);
			if (m_bResiChisq || m_bMDB) m_Qv.mat(nMeas, nMeas);
			if (m_bMDB)                 m_Ry.mat(nMeas, nMeas);

			if (nState != m_nState)
			{
				m_StateXp.mat(nState, 1);
				m_StatePp.mat(nState, nState);
				m_StateX.mat(nState, 1);
				m_StateP.mat(nState, nState);
			}
		}
		else
		{
			m_H.zero();
			m_R.zero();
			m_Inno.zero();
		}
		
		m_nState     = nState;
		m_nMeas      = nMeas;
		m_bPDiagonal = bPDiagonal;
		m_bInit      = true;
		m_InnoChisq  = 0.0;
		m_ResiChisq  = 0.0;
		m_InnoSigma0 = 0.0;
		m_ResiSigma0 = 0.0;
		m_RMS        = 0.0;
		m_bMeasVarPre = false;

		return true;
	}


	void CGLSEstimator::FreeGLS(bool bFreeState)
	{
		if (bFreeState)
		{
			m_StateXp.freeMatrix();
			m_StatePp.freeMatrix();
			m_StateX.freeMatrix();
			m_StateP.freeMatrix();
		}
		
		m_H.freeMatrix();
		m_R.freeMatrix();
		m_Inno.freeMatrix();
		m_Resi.freeMatrix();
		m_Qi.freeMatrix();
		m_Qv.freeMatrix();
		m_Rp.freeMatrix();
		m_K.freeMatrix();
		m_Ry.freeMatrix();
		m_KFTmp_F.freeMatrix();
		m_KFTmp_Q.freeMatrix();
		//lj修改 2026-08-11：释放GNSS区间累计矩阵成员
		m_PStart.freeMatrix();
		m_PhiAcc.freeMatrix();
		m_QAcc.freeMatrix();
		m_AccInitialized = false;
		m_StartNewInterval = true;
		m_HasSuccessfulGNSSBoundary = false;
		m_AccStepCount = 0;
	}


	void CGLSEstimator::ClearGLS()
	{
		m_H.zero();
		m_R.zero();
		m_Inno.zero();
	}

// lj:考虑修改一下这个函数 参数传入GNSSTime 用于输出时间
	void CGLSEstimator::KFPrediction(double dt, const Mat CF, const Mat Qc, const Mat StateP, Mat Phi, Mat StatePPre)
	{
		if (dt <= 0.0)
		{
			StatePPre = StateP;

			// 原来dt不为正数时,表明时离散KF: Pp = F*P*FT + Q
			// 但程序处理过程中,并不能保证dt一定为正数,所以不能用来作为离散KF的标识
			// MatrixMultiply_HPHT(CF.row, CF.col, CF.x, StateP.x, false, StatePPre.x);
			// StatePPre.add2(Qc);
		}
		else
		{
			if (m_KFTmp_F.row != CF.row || m_KFTmp_F.col != CF.col)
			{
				m_KFTmp_F.freeMatrix();
				m_KFTmp_F.mat(CF.row, CF.col);
			}

			if (m_KFTmp_Q.row != Qc.row || m_KFTmp_Q.col != Qc.col)
			{
				m_KFTmp_Q.freeMatrix();
				m_KFTmp_Q.mat(Qc.row, Qc.col);
			}
			
			// lj毕设：这部分需要输出状态转移矩阵和Pp
			// Phi = e^(CF*dt)
			m_KFTmp_F = CF; m_KFTmp_F *= dt;
			Phi.exp(m_KFTmp_F, 2);

			// lj毕设：预测Q阵在此处更换
			// Q = 0.5*((Phi*Qc*Phi_T) + Qc)*dt; 等价三阶模型
			MatrixMultiply_HPHT(Phi.row, Phi.col, Phi.x, Qc.x, false, m_KFTmp_Q.x);
			m_KFTmp_Q.add2(Qc);
			m_KFTmp_Q *= (dt*0.5);

			//lj修改 2026-08-11：使用实例成员累计相邻两次成功GNSS位置更新之间的传播量
			if (!m_AccInitialized || m_PStart.row != StateP.row || m_PStart.col != StateP.col)
			{
				m_PStart.freeMatrix();
				m_PhiAcc.freeMatrix();
				m_QAcc.freeMatrix();
				m_PStart.mat(StateP.row, StateP.col);
				m_PhiAcc.mat(Phi.row, Phi.col);
				m_QAcc.mat(m_KFTmp_Q.row, m_KFTmp_Q.col);
				m_AccInitialized = true;
				m_StartNewInterval = true;
				m_HasSuccessfulGNSSBoundary = false;
			}

			//lj修改 2026-08-11：成功GNSS更新后的第一次传播以更新后的后验协方差为区间起点
			if (m_StartNewInterval)
			{
				m_PStart = StateP;
				m_PhiAcc.zero();
				for (int i = 0; i < m_PhiAcc.row && i < m_PhiAcc.col; i++) m_PhiAcc.at(i, i, 1.0);
				m_QAcc.zero();
				m_AccStepCount = 0;
				m_StartNewInterval = false;
			}

			// Q_acc,new = Phi_current * Q_acc,old * Phi_current^T + Q_current
			Mat QAccNew(m_QAcc.row, m_QAcc.col);
			MatrixMultiply_HPHT(Phi.row, Phi.col, Phi.x, m_QAcc.x, false, QAccNew.x);
			QAccNew.add2(m_KFTmp_Q);
			m_QAcc = QAccNew;

			// Phi_acc,new = Phi_current * Phi_acc,old (order must not be reversed)
			Mat PhiAccNew(m_PhiAcc.row, m_PhiAcc.col);
			PhiAccNew.multiply(Phi, m_PhiAcc);
			m_PhiAcc = PhiAccNew;
			m_AccStepCount++;

			// Pp = Phi*P*Phi_T + Q
			MatrixMultiply_HPHT(Phi.row, Phi.col, Phi.x, StateP.x, false, StatePPre.x);
			StatePPre.add2(m_KFTmp_Q);

		}
	}


	//lj修改 2026-08-11：仅在GNSS位置更新成功后输出并结束当前累计区间
	void CGLSEstimator::FinishGNSSInterval(Mat PPred)
	{
		if (!m_AccInitialized || updateTime == 0) return;

		// 第一次成功GNSS更新仅建立起始边界，不输出启动阶段的不完整区间。
		if (m_HasSuccessfulGNSSBoundary)
		{
			static FILE* fpPStart = fopen("PStart.txt", "w");
			static FILE* fpPhiAcc = fopen("PhiAcc.txt", "w");
			static FILE* fpQAcc = fopen("QAcc.txt", "w");
			static FILE* fpPPredCpp = fopen("PPredCpp.txt", "w");
			static FILE* fpAccMeta = fopen("AccMeta.txt", "w");
			static int headerWritten = 0;

			if (!headerWritten)
			{
				if (fpPStart) fprintf(fpPStart, "WeekSecond PStart\n");
				if (fpPhiAcc) fprintf(fpPhiAcc, "WeekSecond PhiAcc\n");
				if (fpQAcc) fprintf(fpQAcc, "WeekSecond QAcc\n");
				if (fpPPredCpp) fprintf(fpPPredCpp, "WeekSecond PPredCpp\n");
				if (fpAccMeta) fprintf(fpAccMeta, "WeekSecond StepCount\n");
				headerWritten = 1;
			}

			if (fpPStart) { fprintf(fpPStart, "%d", updateTime); m_PStart.write(fpPStart); fprintf(fpPStart, "\n"); }
			if (fpPhiAcc) { fprintf(fpPhiAcc, "%d", updateTime); m_PhiAcc.write(fpPhiAcc); fprintf(fpPhiAcc, "\n"); }
			if (fpQAcc) { fprintf(fpQAcc, "%d", updateTime); m_QAcc.write(fpQAcc); fprintf(fpQAcc, "\n"); }
			if (fpPPredCpp) { fprintf(fpPPredCpp, "%d", updateTime); PPred.write(fpPPredCpp); fprintf(fpPPredCpp, "\n"); }
			if (fpAccMeta) fprintf(fpAccMeta, "%d %d\n", updateTime, m_AccStepCount);
		}

		m_HasSuccessfulGNSSBoundary = true;
		m_StartNewInterval = true;
		updateTime = 0;
	}



	void CGLSEstimator::setXP(vector<int>& index, Mat X, Mat P)
	{
		int i = 0, j = 0, n = index.size();
		for (i = 0; i < n; i++)
		{
			m_StateXp.x[i] = X.x[index[i]];
			for (j = 0; j < m_nState; j++) m_StatePp.pVal(i, j)[0] = P.at(index[i], index[j]);
		}
	}



	void CGLSEstimator::getXP(vector<int>& index, Mat X, Mat P)
	{
		int i = 0, j = 0, n = index.size();
		for (i = 0; i < n; i++)
		{
			X.x[index[i]] = m_StateX.x[i];
			for (j = 0; j < m_nState; j++) P.pVal(index[i], index[j])[0] = m_StateP.at(i, j);
		}
	}


	bool CGLSEstimator::GLS()
	{
		if (m_bInit == false)
		{
			DumpException("Forget to init least squares.", LOGT_INFO | LOGT_AFX, DUMP_LOCATION);
			return false;
		}
		m_bInit = false;

		int i = 0;

		// 对称化
		m_StatePp.symm(); 

		// Rp = H*Pp*HT + R
		// lj毕设：m_H是观测矩阵 需要输出m_H

		MatrixMultiply_HPHT(m_nMeas, m_nState, m_H.x, m_StatePp.x, false, m_Rp.x);
		
		// 观测值预报方差是否大于实际观测方差
		if (m_Rp.trace() > m_R.trace()) m_bMeasVarPre = true; else m_bMeasVarPre = false;
		
		// 对称化	
		// lj毕设：输出m_R
		m_Rp.add2(m_R);
		m_Rp.symm();

		if(m_bInnoChisq) m_Qi = m_Rp;

		if (m_Rp.inv_sp() == false)
		{
			string msg("Fail to perform generalized least squares, the co-factor matrix is singular!");
			DumpException(msg, LOGT_ERROR | LOGT_AFX, DUMP_LOCATION);
			return false;
		}

		// 计算新息的卡方分布值
		if (m_bInnoChisq)
		{
			MatrixMultiply_BTPB(m_nMeas, 1, m_Inno.x, m_Rp.x, false, &m_InnoChisq);
			m_InnoSigma0 = xsqrt(m_InnoChisq / m_nMeas);
		}

		Mat HT(m_nState, m_nMeas);
		HT.transpose(m_H);

		// K = Pp * (HT) * (Rp_1)
		
		m_K.multiply(m_StatePp, HT);
		HT.multiply(m_K, m_Rp);
		m_K = HT;
		HT.freeMatrix();
		
		// X = Xp + K * Inno
		// lj毕设：输出m_Inno
		// lj毕设：m_StateX
		m_StateX.multiply(m_K, m_Inno);
		m_dXl = m_StateX.norm2();
		m_StateX.add2(m_StateXp);

		// Joseph 形式, P = (I-KH)*Pp(I-KH)T + K*R*KT

		// M = I - K*H
		Mat M(m_nState, m_nState);
		M.multiply(m_K,m_H);
		for (i = 0; i < m_nState; i++) M.x[i*m_nState + i] -= 1.0;
		M.negative(); // 取反

		// (I-KH)*Pp(I-KH)T
		MatrixMultiply_HPHT(m_nState, m_nState, M.x, m_StatePp.x, false, m_StateP.x);

		// P = K * R * KT;
		MatrixMultiply_HPHT(m_nState, m_nMeas, m_K.x, m_R.x, m_bPDiagonal, M.x);
		m_StateP.add2(M); m_StateP.symm();

		M.freeMatrix();

		// 检查P对角线是否小于0
		for (i = 0; i < m_nState; i++)
		{
			if (m_StateP.x[i*m_nState+i] <= 0.0)
			{
				//printf("%d %.15f\n", i, m_StateP.x[i*m_nState + i]);
				string msg("The diagonal element of state variance is not positive.");
				DumpException(msg, LOGT_ERROR | LOGT_AFX, DUMP_LOCATION);
				return false;
			}
		}
		

		if (m_bResiChisq || m_bMDB)
		{
			MatrixMultiply_HPHT(m_nMeas, m_nState, m_H.x, m_StateP.x, false, m_Qv.x);
			m_Qv.symm();

			// 状态更新完成以后,重新计算新息
			// 此时的新息就是验后残差,其方差就是R+H*P*H
			// 注意,这里其实计算的是下一次迭代的新息方差,而不是验后残差的方差
			// 这里的方差实践中可以设为三种: R+H*P*H, R-H*P*H , R. 都可以
			// 因此最终得到的是验后残差的sigma0
			if (m_bResiChisq)
			{
				// v = Inno - H*(X - Xp)
				Mat dX(m_nState, 1);
				Mat Qv(m_Qv.row, m_Qv.col);
				Qv.add(m_Qv, m_R);
				Qv.inv_sp();

				dX.subtract(m_StateX, m_StateXp);
				m_Resi.multiply(m_H, dX);
				m_Resi.subtract_l(m_Inno);

				// 计算残差的卡方分布值
				MatrixMultiply_BTPB(m_nMeas, 1, m_Resi.x, Qv.x, false, &m_ResiChisq);

				m_ResiSigma0 = xsqrt(m_ResiChisq / m_nMeas);

				m_RMS = xsqrt(MatrixDot(m_nMeas, 1, m_Resi.x, m_Resi.x) / m_nMeas);

				dX.freeMatrix();
				Qv.freeMatrix();
			}

			// (R - H * P * HT) 得到验后残差的方差
			m_Qv.subtract_l(m_R);

			// Ry = (R - H * P * HT)*inv(R) = Qv*inv(R), 多余观测分量,计算MDB时该项需要用到
			if (m_bMDB)
			{
				m_Rp = m_R; m_Rp.inv_sp();
				m_Ry.multiply(m_Qv, m_Rp);
			}
		}

		// 调试输出用
		// lj毕设 参考这里 如果找不到了StateX就是t时刻输出t时刻的 下个时刻调用上一时刻的
		if (0)
		{
			FILE* fp = fopen("D:\\GLS2.txt", "w");
			m_StateXp.write(fp);
			m_StatePp.write(fp);
			// lj毕设：这三个
			m_H.write(fp);
			m_R.write(fp);
			m_Inno.write(fp);
			// 用于下一时刻 那只要在外部输出一个 状态转移矩阵+IMU速度+传感器数据 即可
			m_StateX.write(fp);
			m_StateP.write(fp);
			fclose(fp);
		}

		return true;
	}









	CGLSEstimatorX::CGLSEstimatorX()
	{
		m_bInit = false;
		m_bMDB = true;
		m_bPDiagonal = false;
		m_bInnoChisq = true;
		m_bResiChisq = true;
		m_nState = 0;
		m_nMeas = 0;
	}


	CGLSEstimatorX::~CGLSEstimatorX()
	{
		FreeGLS(true);
	}


	bool CGLSEstimatorX::initGLS(int nState, int nMeas, bool bPDiagonal)
	{
		if (nState == 0 || nMeas == 0) return false;

		if (nMeas != m_nMeas || nState != m_nState)
		{
			FreeGLS(nState != m_nState);
			m_H.mat(nMeas, nState);
			m_R.mat(nMeas, nMeas);
			m_Inno.mat(nMeas, 1);
			if (m_bResiChisq)           m_Resi.mat(nMeas, 1);
			if (m_bInnoChisq)           m_Qi.mat(nMeas, nMeas);
			if (m_bResiChisq || m_bMDB) m_Qv.mat(nMeas, nMeas);
			if (m_bMDB)                 m_Ry.mat(nMeas, nMeas);

			if (nState != m_nState)
			{
				m_StateXp.mat(nState, 1);
				m_StatePp.mat(nState, nState);
				m_StateX.mat(nState, 1);
				m_StateP.mat(nState, nState);

				m_SRKF_K = MatrixNew(nState, 1);
				m_SRKF_a = MatrixNew(nState, 1);
			}

			if (bPDiagonal == false)
			{
				m_HTmp.mat(nMeas, nState);
			}
			m_InnoTmp.mat(nMeas, 1);
			m_RpTmp.mat(nMeas, nMeas);
			m_Rp_S = MatrixNew(nMeas*(nMeas + 1) / 2, 1);
			m_R_S = MatrixNew(nMeas*(nMeas + 1) / 2, 1);
		}
		else
		{
			m_H.zero();
			m_R.zero();
			m_Inno.zero();
		}

		m_nState     = nState;
		m_nMeas      = nMeas;
		m_bPDiagonal = bPDiagonal;
		m_bInit      = true;
		m_InnoChisq  = 0.0;
		m_ResiChisq  = 0.0;
		m_InnoSigma0 = 0.0;
		m_ResiSigma0 = 0.0;
		m_RMS        = 0.0;
		m_bMeasVarPre = false;

		return true;
	}


	void CGLSEstimatorX::FreeGLS(bool bFreeState)
	{
		if (bFreeState)
		{
			m_StateXp.freeMatrix();
			m_StatePp.freeMatrix();
			m_StateX.freeMatrix();
			m_StateP.freeMatrix();

			if (m_SRKF_K) free(m_SRKF_K);
			if (m_SRKF_a) free(m_SRKF_a);
		}

		m_H.freeMatrix();
		m_R.freeMatrix();
		m_Inno.freeMatrix();
		m_Resi.freeMatrix();
		m_Qi.freeMatrix();
		m_Qv.freeMatrix();
		m_Ry.freeMatrix();
		m_KFTmp_F.freeMatrix();
		m_KFTmp_Q.freeMatrix();
		m_HTmp.freeMatrix();
		m_InnoTmp.freeMatrix();
		m_RpTmp.freeMatrix();
		if (m_Rp_S) free(m_Rp_S);
		if (m_R_S) free(m_R_S);
	}


	void CGLSEstimatorX::ClearGLS()
	{
		m_H.zero();
		m_R.zero();
		m_Inno.zero();
	}


	void CGLSEstimatorX::KFPrediction(double dt, const Mat CF, const Mat Qc, const Mat StateP, Mat Phi, Mat StatePPre)
	{
		if (dt <= 0.0)
		{
			StatePPre = StateP;

			// 原来dt不为正数时,表明时离散KF: Pp = F*P*FT + Q
			// 但程序处理过程中,并不能保证dt一定为正数,所以不能用来作为离散KF的标识
			// MatrixMultiply_HPHT(CF.row, CF.col, CF.x, StateP.x, false, StatePPre.x);
			// StatePPre.add2(Qc);
		}
		else
		{
			if (m_KFTmp_F.row != CF.row || m_KFTmp_F.col != CF.col)
			{
				m_KFTmp_F.freeMatrix();
				m_KFTmp_F.mat(CF.row, CF.col);
			}

			if (m_KFTmp_Q.row != Qc.row || m_KFTmp_Q.col != Qc.col)
			{
				m_KFTmp_Q.freeMatrix();
				m_KFTmp_Q.mat(Qc.row, Qc.col);
			}

			// Phi = e^(CF*dt)
			m_KFTmp_F = CF; m_KFTmp_F *= dt;
			Phi.exp(m_KFTmp_F, 2);

			// Q = 0.5*(Phi*Qc*Phi_T) + Qc)*dt;
			MatrixMultiply_HPHT(Phi.row, Phi.col, Phi.x, Qc.x, false, m_KFTmp_Q.x);
			m_KFTmp_Q.add2(Qc);
			m_KFTmp_Q *= (dt*0.5);

			// Pp = Phi*P*Phi_T + Q
			MatrixMultiply_HPHT(Phi.row, Phi.col, Phi.x, StateP.x, false, StatePPre.x);
			StatePPre.add2(m_KFTmp_Q);
		}
	}


	void CGLSEstimatorX::setXP(vector<int>& index, Mat X, Mat P)
	{
		int i = 0, j = 0, n = index.size();
		for (i = 0; i < n; i++)
		{
			m_StateXp.x[i] = X.x[index[i]];
			for (j = 0; j < m_nState; j++) m_StatePp.pVal(i, j)[0] = P.at(index[i], index[j]);
		}
	}


	void CGLSEstimatorX::getXP(vector<int>& index, Mat X, Mat P)
	{
		int i = 0, j = 0, n = index.size();
		for (i = 0; i < n; i++)
		{
			X.x[index[i]] = m_StateX.x[i];
			for (j = 0; j < m_nState; j++) P.pVal(index[i], index[j])[0] = m_StateP.at(i, j);
		}
	}


	bool CGLSEstimatorX::GLS1_EKF()
	{
		int i, j, k;

		if (m_StatePp.chold() == false)
		{
			DumpException("Cholesky decomposition is failed.", LOGT_ERROR | LOGT_AFX, DUMP_LOCATION);
			return false;
		}

		double b = 0.0;
		double r = 0.0;
		double R = 0.0;
		double v = 0.0;

		// 平方根滤波的Potter算法迭代
		for (j = 0; j < m_nMeas; j++)
		{
			R = 1.0; // R已经处理过了,变成单位阵
			if (m_bPDiagonal) R = m_R.at(j, j);

			// 计算a
			double* Hj = m_H.x + j*m_H.col;
			MatrixMultiply(1, m_nState, Hj, m_nState, m_nState, m_StatePp.x, m_SRKF_a);

			// 计算b
			b = 1.0 / (MatrixDot(m_nState, m_SRKF_a, m_SRKF_a) + R);

			// 计算r
			r = 1.0 / (1 + xsqrt(b*R));

			// 计算K
			MatrixMultiply(m_nState, m_nState, m_StatePp.x, m_nState, 1, m_SRKF_a, m_SRKF_K, b);

			// 计算X
			v = m_Inno.x[j] - MatrixDot(m_nState, Hj, m_StateX.x);
			for (i = 0; i < m_nState; i++) m_StateX.x[i] += m_SRKF_K[i] * v;

			// 计算P
			for (i = 0; i < m_nState; i++)
				for (k = 0; k < m_nState; k++)
					m_StatePp.pVal(i, k)[0] -= r*m_SRKF_K[i] * m_SRKF_a[k];
		}

		if (m_bResiChisq)
		{
			m_Resi.multiply(m_H, m_StateX);
			m_Resi.subtract_l(m_Inno);
		}
		
		// 得到估计值
		m_dXl = m_StateX.norm2();
		m_StateX.add2(m_StateXp);
		MatrixMultiply_HPHT(m_nState, m_nState, m_StatePp.x, NULL, true, m_StateP.x);

		return true;
	}


	bool CGLSEstimatorX::GLS2_EKF()
	{
		int i, j, k;

		if (m_StatePp.chold() == false)
		{
			DumpException("Cholesky decomposition is failed.", LOGT_ERROR | LOGT_AFX, DUMP_LOCATION);
			return false;
		}

		if (m_bPDiagonal == false)
		{
			// 得到R的平方根分解
			if (MatrixCholesky(m_nMeas, m_nMeas, m_R.x, m_R_S) == false) return false;
			MatrixCholeskyInv(m_nMeas, m_nMeas, m_R_S, true);

			// 得到新的H和Inno
			MatrixMultiply_LTXM(m_nMeas, m_R_S, m_H.row, m_H.col, m_H.x, m_HTmp.x);
			MatrixMultiply_LTXM(m_nMeas, m_R_S, m_Inno.row, m_Inno.col, m_Inno.x, m_InnoTmp.x);
		}

		double b = 0.0;
		double r = 0.0;
		double R = 0.0;
		double v = 0.0;

		// 平方根滤波的Potter算法迭代
		for (j = 0; j < m_nMeas; j++)
		{
			R = 1.0; // R已经处理过了,变成单位阵
			if (m_bPDiagonal) R = m_R.at(j, j);

			// 计算a
			double* Hj = m_HTmp.x + j*m_HTmp.col;
			MatrixMultiply(1, m_nState, Hj, m_nState, m_nState, m_StatePp.x, m_SRKF_a);

			// 计算b
			b = 1.0 / (MatrixDot(m_nState, m_SRKF_a, m_SRKF_a) + R);

			// 计算r
			r = 1.0 / (1 + xsqrt(b*R));

			// 计算K
			MatrixMultiply(m_nState, m_nState, m_StatePp.x, m_nState, 1, m_SRKF_a, m_SRKF_K, b);

			// 计算X
			v = m_InnoTmp.x[j] - MatrixDot(m_nState, Hj, m_StateX.x);
			for (i = 0; i < m_nState; i++) m_StateX.x[i] += m_SRKF_K[i] * v;

			// 计算P
			for (i = 0; i < m_nState; i++)
				for (k = 0; k < m_nState; k++)
					m_StatePp.pVal(i, k)[0] -= r*m_SRKF_K[i] * m_SRKF_a[k];
		}

		if (m_bResiChisq)
		{
			m_Resi.multiply(m_H, m_StateX);
			m_Resi.subtract_l(m_Inno);
		}

		// 得到估计值
		m_dXl = m_StateX.norm2();
		m_StateX.add2(m_StateXp);
		MatrixMultiply_HPHT(m_nState, m_nState, m_StatePp.x, NULL, true, m_StateP.x);

		return true;
	}


	bool CGLSEstimatorX::GLS1_LSQ()
	{
		int i, j, k;
		double Sum;

		// 1. 计算HT*inv(R)*H
		for (i = 0; i < m_nState; i++)
		{
			for (j = 0; j < m_nState; j++)
			{
				Sum = 0.0;

				for (k = 0; k < m_nMeas; k++)
				{
					Sum += m_H.x[k*m_nState + i] * m_H.x[k*m_nState + j] / m_R.x[k*m_nMeas + k];
				}

				m_StateP.x[i*m_nState + j] = Sum;
			}
		}

		// 2. 计算状态协方差矩阵
		if (m_StatePp.inv_sp() == false) return false;
		m_StateP.add2(m_StatePp);
		if (m_StateP.inv_sp() == false) return false;

		// 3. 计算HT*inv(R)*v
		Mat W; W.mat(m_nState, 1);
		for (i = 0; i < m_nState; i++)
		{
			Sum = 0;

			for (k = 0; k < m_nMeas; k++)
			{
				Sum += m_H.x[k*m_nState + i] * m_Inno.x[k] / m_R.x[k*m_nMeas + k];
			}

			W.x[i] = Sum;
		}

		// 4. 得到最终结果
		m_StateX.multiply(m_StateP, W);
		m_dXl = m_StateX.norm2();
		m_StateX.add2(m_StateXp);

		W.freeMatrix();

		return true;
	}


	bool CGLSEstimatorX::GLS2_LSQ()
	{
		int i, k;
		double Sum;

		// 1. 计算HT*inv(R)*H
		if (MatrixCholesky(m_nMeas, m_nMeas, m_R.x, m_R_S) == false) return false;
		if (MatrixCholeskyInv(m_nMeas, m_nMeas, m_R_S, true) == false) return false;
		MatrixMultiply_LTXM(m_nMeas, m_R_S, m_H.row, m_H.col, m_H.x, m_HTmp.x);
		MatrixMultiply_BTPB(m_nMeas, m_nState, m_HTmp.x, NULL, true, m_StateP.x);

		// 2. 计算状态协方差矩阵
		if (m_StatePp.inv_sp() == false)return false;
		m_StateP.add2(m_StatePp);
		if (m_StateP.inv_sp() == false)return false;

		MatrixMultiply_LTXM(m_nMeas, m_R_S, m_Inno.row, m_Inno.col, m_Inno.x, m_InnoTmp.x);
		
		// 3. 计算HT*inv(R)*v
		Mat W; W.mat(m_nState, 1);
		for (i = 0; i < m_nState; i++)
		{
			Sum = 0;

			for (k = 0; k < m_nMeas; k++)
			{
				Sum += m_HTmp.x[k*m_nState + i] * m_InnoTmp.x[k];
			}

			W.x[i] = Sum;
		}

		// 4. 得到最终结果
		m_StateX.multiply(m_StateP, W);
		m_dXl = m_StateX.norm2();
		m_StateX.add2(m_StateXp);

		W.freeMatrix();

		return true;
	}


#define GLS_DEBUG_OUT_

	// 速度有待进一步提升
	bool CGLSEstimatorX::GLS()
	{

#ifdef GLS_DEBUG_OUT

		FILE* fp = fopen("D:\\GLS.txt", "w");
		m_StateXp.write(fp);
		m_StatePp.write(fp);
		m_H.write(fp);
		m_R.write(fp);
		m_Inno.write(fp);

#endif // GLS_DEBUG_OUT

		if (m_bInit == false)
		{
			DumpException("Forget to init generalized least squares.", LOGT_INFO | LOGT_AFX, DUMP_LOCATION);
			return false;
		}

		m_bInit = false;

		m_StateX.zero();   // 置零
		m_StatePp.symm();  // 对称化

		// 1. 计算新息的卡方分布和Sigma0
		if (m_bInnoChisq)
		{
			// Rp = H*Pp*HT + R
			m_bMeasVarPre = MatrixMultiply_HPHTR(m_nMeas, m_nState, m_H.x, m_StatePp.x, false, m_R.x, m_Qi.x);
			
			if (MatrixCholesky(m_nMeas, m_nMeas, m_Qi.x, m_Rp_S))
			{
				MatrixCholeskyInv(m_nMeas, m_nMeas, m_Rp_S, true);
				MatrixMultiply_LTXM(m_nMeas, m_Rp_S, m_nMeas, 1, m_Inno.x, m_InnoTmp.x);
				m_InnoChisq = MatrixDot(m_nMeas, m_InnoTmp.x, m_InnoTmp.x);
				m_InnoSigma0 = xsqrt(m_InnoChisq / m_nMeas);
			}
		}


		// 2. 根据维数开始KF滤波或LS最小二乘
		bool bSuc = true;
		if (m_nMeas > m_nState)
		{
			if (m_bPDiagonal) bSuc = GLS1_LSQ(); else bSuc = GLS2_LSQ();
		}
		else
		{
			if (m_bPDiagonal) bSuc = GLS1_EKF(); else bSuc = GLS2_EKF();
		}
		if (bSuc == false) return false;


		// 3. 检查P对角线是否小于0
		for (int i = 0; i < m_nState; i++)
		{
			if (m_StateP.x[i*m_nState + i] <= 0.0)
			{
				//char msg[1024] = { '\0' };
				//sprintf(msg, "StateP < 0.0, index:%d, val:%.15f", i, m_StateP.x[i*m_nState + i]);
				string msg("The diagonal element of state variance is not positive.");
				DumpException(msg, LOGT_ERROR | LOGT_AFX, DUMP_LOCATION);
				return false;
			}
		}

		// 4. 计算验后残差: v = Inno - H*(X - Xp)
		Mat dX(m_nState, 1);
		Mat Qv(m_Qv.row, m_Qv.col);
		Qv.add(m_Qv, m_R);
		Qv.inv_sp();

		dX.subtract(m_StateX, m_StateXp);
		m_Resi.multiply(m_H, dX);
		m_Resi.subtract_l(m_Inno);

		m_RMS = xsqrt(MatrixDot(m_nMeas, 1, m_Resi.x, m_Resi.x) / m_nMeas);

		// 4. 计算验后残差的卡方分布和Sigma0
		if (m_bResiChisq)
		{
			// 状态更新完成以后,重新计算新息
			// 此时的新息就是验后残差,其方差就是R+H*P*H
			// 注意,这里其实计算的是下一次迭代的新息方差,而不是验后残差的方差
			// 这里的方差实践中可以设为三种: R+H*P*H, R-H*P*H , R. 都可以
			// 因此最终得到的是验后残差的sigma0
			MatrixMultiply_HPHTR(m_nMeas, m_nState, m_H.x, m_StateP.x, false, m_R.x, m_RpTmp.x);

			if (MatrixCholesky(m_nMeas, m_nMeas, m_RpTmp.x, m_Rp_S))
			{
				MatrixCholeskyInv(m_nMeas, m_nMeas, m_Rp_S, true);
				MatrixMultiply_LTXM(m_nMeas, m_Rp_S, m_nMeas, 1, m_Resi.x, m_InnoTmp.x);
				m_ResiChisq = MatrixDot(m_nMeas, m_InnoTmp.x, m_InnoTmp.x);
				m_ResiSigma0 = xsqrt(m_ResiChisq / m_nMeas);
			}
		}
		

		// 5. 得到多余观测分量矩阵
		if (m_bMDB && m_bResiChisq)
		{
			MatrixSubtraction(m_nMeas, m_nMeas, m_R.x, m_RpTmp.x, m_Qv.x, 2.0, 1.0);

			if (m_bPDiagonal)
			{
				for (int i = 0; i < m_nMeas; i++)
					for (int j = 0; j < m_nMeas; j++)
					{
						m_Ry.pVal(j, i)[0] = m_Qv.at(j, i) / m_R.at(i, i);
					}
			}
			else
			{
				// m_R_S在GLS2()中有R分解得到的,这里操作后得到inv(R) 
				MatrixMultiply_LTTXLT(m_nMeas, m_nMeas, m_R_S, m_RpTmp.x);
				m_Ry.multiply(m_Qv, m_RpTmp);
			}
		}


#ifdef GLS_DEBUG_OUT
		m_StateX.write(fp);
		m_StateP.write(fp);
		fclose(fp);
#endif

		return true;
	}
}
