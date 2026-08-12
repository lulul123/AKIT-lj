#include "SINSDLL/CSINSApplication.h"

#include <algorithm>

#include "BASEDLL/BaseApp.h"
#include "BASEDLL/CPose3D.h"
#include "BASEDLL/BaseTime.h"
#include "BASEDLL/BaseMath.h"
#include "BASEDLL/BaseCmnFunc.h"
#include "BASEDLL/Smoother.h"
#include "BASEDLL/GoogleEarth.h"

#include "GNSSDLL/CGNSSDataConvert.h"
#include "GNSSDLL/GNSSCmnFunc.h"
#include "GNSSDLL/CRTKPoint.h"
#include "GNSSDLL/CSwitchVRS.h"

#include "SINSDLL/SINSCmnFunc.h"
#include "SINSDLL/CSINSDataConvert.h"
#include "SINSDLL/CSINSAlignment.h"
#include "SINSDLL/CLooseCoupling.h"
#include "SINSDLL/CPPPTightCoupling.h"
#include "SINSDLL/CRTKTightCoupling.h"
#include "SINSDLL/CLCPAD.h"
#include "SINSDLL/CSINSInitState.h"
#include "SINSDLL/CSINSInstallation.h"

#include <time.h> // ITS 文章 计算耗时

namespace sinstk
{
	
#define _TwiceProc_LC_ON_



	// ************************************************************************************************************************************** //
	//  previous class:  ;     next class : CSINSApplication                                                                                  //
	// ************************************************************************************************************************************** //


	static bool ls_bCheckScene = false;
	static int  ls_nCheckPass = 0;


	CSINSApplication::CSINSApplication()
	{
		m_pGNSSOpt   = NULL;
		m_AppInfo    = NULL;
		m_IMUGPSWeek = -1;
		m_bMultiBaseSite = false;
	}


	CSINSApplication::~CSINSApplication()
	{
		
	}


	bool CSINSApplication::init()
	{
		setOption();

		m_SINSOpt.init();

		if(getAllFiles() == false) return false;

		GetProgramDir();

		return true;
	}


	void CSINSApplication::setOption()
	{
		m_pGNSSOpt = &m_SINSOpt.m_GNSSOpt;

		char title[1024] = { '\0' };
		getFileInfo(m_pGNSSOpt->m_OBSFile[0].c_str(), title, NULL, NULL, NULL);

		string Suffix = "";
		if (m_pGNSSOpt->m_FilterType == GNSSFILTER_Forward)       Suffix = ".bffp";
		else if (m_pGNSSOpt->m_FilterType == GNSSFILTER_Reverse)  Suffix = ".brfp";
		else if (m_pGNSSOpt->m_FilterType >= GNSSFILTER_Combined) Suffix = ".bdfp";

		if(m_pGNSSOpt->m_bProjPath)
		{
			for (int i = 0; i < MAXBASESITE; i++)
			{
				m_OBSFile[i] = m_pGNSSOpt->m_OBSFile[i];
				
				for (int k = 0; k <= NSYS; k++)
				{
					m_NavEphFile[k][i] = m_pGNSSOpt->m_ProjPath + FILEPATHSEP + "I_OBS_FILE" + FILEPATHSEP + m_pGNSSOpt->m_NavEphFile[k][i];
				}
			}

			for (int k = 0; k < NSYS; k++)
			{
				m_PreEphFile[k] = m_pGNSSOpt->m_ProjPath + FILEPATHSEP + "I_IGS_FILE" + FILEPATHSEP + m_pGNSSOpt->m_PreEphFile[k];
				m_PreClkFile[k] = m_pGNSSOpt->m_ProjPath + FILEPATHSEP + "I_IGS_FILE" + FILEPATHSEP + m_pGNSSOpt->m_PreClkFile[k];
			}

			m_BLQFile = m_pGNSSOpt->m_ProjPath + FILEPATHSEP + "I_IGS_FILE" + FILEPATHSEP + m_pGNSSOpt->m_BLQFile;
			m_FCBFile = m_pGNSSOpt->m_ProjPath + FILEPATHSEP + "I_IGS_FILE" + FILEPATHSEP + m_pGNSSOpt->m_PPPAR_FCBFile;
			m_IMUFile = m_pGNSSOpt->m_ProjPath + FILEPATHSEP + "I_IMU_FILE" + FILEPATHSEP + m_SINSOpt.m_IMUFile;
			m_POSFile = m_pGNSSOpt->m_ProjPath + FILEPATHSEP + "I_ELC_FILE" + FILEPATHSEP + m_SINSOpt.m_POSFile;
			m_DMIFile = m_pGNSSOpt->m_ProjPath + FILEPATHSEP + "I_ELC_FILE" + FILEPATHSEP + m_SINSOpt.m_DMIFile;
			m_HEDFile = m_pGNSSOpt->m_ProjPath + FILEPATHSEP + "I_ELC_FILE" + FILEPATHSEP + m_SINSOpt.m_HEDFile;
			m_ZUPFile = m_pGNSSOpt->m_ProjPath + FILEPATHSEP + "I_ELC_FILE" + FILEPATHSEP + m_SINSOpt.m_ZUPFile;
			m_ZIHFile = m_pGNSSOpt->m_ProjPath + FILEPATHSEP + "I_ELC_FILE" + FILEPATHSEP + m_SINSOpt.m_ZIHFile;
			m_NHCFile = m_pGNSSOpt->m_ProjPath + FILEPATHSEP + "I_ELC_FILE" + FILEPATHSEP + m_SINSOpt.m_NHCFile;
			m_CUPFile = m_pGNSSOpt->m_ProjPath + FILEPATHSEP + "I_ELC_FILE" + FILEPATHSEP + m_SINSOpt.m_CUPFile;

			// 如果IPS文件是空的
			if (m_SINSOpt.m_GNSSFile == "")
			{
				if (m_pGNSSOpt->m_PMode == GNSSMODE_SPP)
				{
					m_GNSSFile = m_pGNSSOpt->m_ProjPath + FILEPATHSEP + "R_SPP" + FILEPATHSEP + string(title) + Suffix;
				}
				else if (m_pGNSSOpt->m_PMode == GNSSMODE_PPP_STATIC || m_pGNSSOpt->m_PMode == GNSSMODE_PPP_KINEMA || m_pGNSSOpt->m_PMode == GNSSMODE_PPP_LEO)
				{
					m_GNSSFile = m_pGNSSOpt->m_ProjPath + FILEPATHSEP + "R_PPP" + FILEPATHSEP + string(title) + Suffix;
				}
				else if (m_pGNSSOpt->m_PMode == GNSSMODE_RTD || m_pGNSSOpt->m_PMode == GNSSMODE_RTK_KINEMA || m_pGNSSOpt->m_PMode == GNSSMODE_RTK_STATIC)
				{
					m_GNSSFile = m_pGNSSOpt->m_ProjPath + FILEPATHSEP + "R_RTK" + FILEPATHSEP + string(title) + Suffix;
				}
				else if (m_pGNSSOpt->m_PMode == GNSSMODE_RTK_MOVEB)
				{
					m_GNSSFile = m_pGNSSOpt->m_ProjPath + FILEPATHSEP + "R_MOV" + FILEPATHSEP + string(title) + Suffix;
				}
				else
				{
					if (m_SINSOpt.m_PMode > 1)
					{
						DumpException("GNSS process mode error!", LOGT_WARNING | LOGT_AFX, DUMP_LOCATION);
					}
				}
			}
			
			return;
		}

		for (int i = 0; i < MAXBASESITE; i++)
		{
			m_OBSFile[i]    = m_pGNSSOpt->m_OBSFile[i];

			for (int k = 0; k <= NSYS; k++)
			{
				m_NavEphFile[k][i] = m_pGNSSOpt->m_NavEphFile[k][i];
			}
		}

		for (int k = 0; k < NSYS; k++)
		{
			m_PreEphFile[k] = m_pGNSSOpt->m_PreEphFile[k];
			m_PreClkFile[k] = m_pGNSSOpt->m_PreClkFile[k];
		}

		m_BLQFile = m_pGNSSOpt->m_BLQFile;
		m_FCBFile = m_pGNSSOpt->m_PPPAR_FCBFile;
		m_IMUFile = m_SINSOpt.m_IMUFile;
		m_GNSSFile = m_SINSOpt.m_GNSSFile;
		m_POSFile = m_SINSOpt.m_POSFile;
		m_DMIFile = m_SINSOpt.m_DMIFile;
		m_HEDFile = m_SINSOpt.m_HEDFile;
		m_ZUPFile = m_SINSOpt.m_ZUPFile;
		m_ZIHFile = m_SINSOpt.m_ZIHFile;
		m_NHCFile = m_SINSOpt.m_NHCFile;
		m_CUPFile = m_SINSOpt.m_CUPFile;

		// 如果IPS文件是空的
		if (m_SINSOpt.m_GNSSFile == "")
		{
			if (m_pGNSSOpt->m_PMode == GNSSMODE_SPP)
			{
				m_GNSSFile = m_pGNSSOpt->m_GNSS_ResultPath + "R_SPP" + FILEPATHSEP + string(title) + Suffix;
			}
			else if (m_pGNSSOpt->m_PMode == GNSSMODE_PPP_STATIC || m_pGNSSOpt->m_PMode == GNSSMODE_PPP_KINEMA || m_pGNSSOpt->m_PMode == GNSSMODE_PPP_LEO)
			{
				m_GNSSFile = m_pGNSSOpt->m_GNSS_ResultPath + "R_PPP" + FILEPATHSEP + string(title) + Suffix;
			}
			else if (m_pGNSSOpt->m_PMode == GNSSMODE_RTD || m_pGNSSOpt->m_PMode == GNSSMODE_RTK_KINEMA || m_pGNSSOpt->m_PMode == GNSSMODE_RTK_STATIC)
			{
				m_GNSSFile = m_pGNSSOpt->m_GNSS_ResultPath + "R_RTK" + FILEPATHSEP + string(title) + Suffix;
			}
			else if (m_pGNSSOpt->m_PMode == GNSSMODE_RTK_MOVEB)
			{
				m_GNSSFile = m_pGNSSOpt->m_GNSS_ResultPath + "R_MOV" + FILEPATHSEP + string(title) + Suffix;
			}
			else
			{
				if (m_SINSOpt.m_PMode > 1)
				{
					DumpException("GNSS process mode error!", LOGT_WARNING | LOGT_AFX, DUMP_LOCATION);
				}
			}
		}
	}


	bool CSINSApplication::getAllFiles()
	{
		char cFileTitle[1024] = { '\0' };
		char cFileName[1024] = { '\0' };
		char cDir[1024] = { '\0' };

		if (!getFileInfo(_C(m_OBSFile[0].c_str()), cFileTitle, NULL, NULL, cDir) || !getFileInfo(_C(m_IMUFile.c_str()), NULL, NULL, cFileName, NULL))
		{
			string msg("Fail to get the observation file information!");
			DumpException(msg, LOGT_WARNING | LOGT_AFX, DUMP_LOCATION);
			return false;
		}

		string FileTitle = string(cFileTitle);
		string FileName = string(cFileName);

		if (m_pGNSSOpt->m_bProjPath)
		{
			m_TRJFile = m_IMUFile + ".ref";

			if (m_SINSOpt.m_PMode == 1)
			{
				m_FilePath = m_pGNSSOpt->m_ProjPath + FILEPATHSEP + "R_INS";
			}
			else if (m_SINSOpt.m_PMode == 2)
			{
				if (m_pGNSSOpt->m_PMode == GNSSMODE_SPP)
				{
					m_FilePath = m_pGNSSOpt->m_ProjPath + FILEPATHSEP + "R_SPP_LCI";
				}
				else if (m_pGNSSOpt->m_PMode == GNSSMODE_PPP_STATIC || m_pGNSSOpt->m_PMode == GNSSMODE_PPP_KINEMA || m_pGNSSOpt->m_PMode == GNSSMODE_PPP_LEO)
				{
					m_FilePath = m_pGNSSOpt->m_ProjPath + FILEPATHSEP + "R_PPP_LCI";
				}
				else if (m_pGNSSOpt->m_PMode == GNSSMODE_RTD || m_pGNSSOpt->m_PMode == GNSSMODE_RTK_KINEMA || m_pGNSSOpt->m_PMode == GNSSMODE_RTK_STATIC)
				{
					m_FilePath = m_pGNSSOpt->m_ProjPath + FILEPATHSEP + "R_RTK_LCI";
				}
				else
				{
					if (m_SINSOpt.m_PMode > 1)
					{
						DumpException("GNSS process mode error!", LOGT_WARNING | LOGT_AFX, DUMP_LOCATION);
						return false;
					}
				}
			}
			else if (m_SINSOpt.m_PMode == 3)
			{
				if (m_pGNSSOpt->m_PMode == GNSSMODE_SPP)
				{
					m_FilePath = m_pGNSSOpt->m_ProjPath + FILEPATHSEP + "R_SPP_TCI";
				}
				else if (m_pGNSSOpt->m_PMode == GNSSMODE_PPP_STATIC || m_pGNSSOpt->m_PMode == GNSSMODE_PPP_KINEMA || m_pGNSSOpt->m_PMode == GNSSMODE_PPP_LEO)
				{
					m_FilePath = m_pGNSSOpt->m_ProjPath + FILEPATHSEP + "R_PPP_TCI";
				}
				else if (m_pGNSSOpt->m_PMode == GNSSMODE_RTD || m_pGNSSOpt->m_PMode == GNSSMODE_RTK_KINEMA || m_pGNSSOpt->m_PMode == GNSSMODE_RTK_STATIC)
				{
					m_FilePath = m_pGNSSOpt->m_ProjPath + FILEPATHSEP + "R_RTK_TCI";
				}
				else
				{
					DumpException("GNSS process mode error!", LOGT_WARNING | LOGT_AFX, DUMP_LOCATION);
					return false;
				}
			}
			else if (m_SINSOpt.m_PMode == 4)
			{
				m_FilePath = m_pGNSSOpt->m_ProjPath + FILEPATHSEP + "R_PAD";
			}
			else if (m_SINSOpt.m_PMode == 5)
			{
				if (m_pGNSSOpt->m_PMode == GNSSMODE_SPP)
				{
					m_FilePath = m_pGNSSOpt->m_ProjPath + FILEPATHSEP + "R_SPP_STC";
				}
				else if (m_pGNSSOpt->m_PMode == GNSSMODE_PPP_STATIC || m_pGNSSOpt->m_PMode == GNSSMODE_PPP_KINEMA || m_pGNSSOpt->m_PMode == GNSSMODE_PPP_LEO)
				{
					m_FilePath = m_pGNSSOpt->m_ProjPath + FILEPATHSEP + "R_PPP_STC";
				}
				else if (m_pGNSSOpt->m_PMode == GNSSMODE_RTD || m_pGNSSOpt->m_PMode == GNSSMODE_RTK_KINEMA || m_pGNSSOpt->m_PMode == GNSSMODE_RTK_STATIC)
				{
					m_FilePath = m_pGNSSOpt->m_ProjPath + FILEPATHSEP + "R_RTK_STC";
				}
				else
				{
					DumpException("GNSS process mode error!", LOGT_WARNING | LOGT_AFX, DUMP_LOCATION);
					return false;
				}
			}
		}
		else
		{
			m_FilePath = string(cDir);

			if (m_SINSOpt.m_SINS_ResultPath == "") m_FilePath = string(cDir) + FILEPATHSEP;
			else m_FilePath = m_SINSOpt.m_SINS_ResultPath;

			m_TRJFile = m_FilePath + FileName + ".ref";

			if (m_SINSOpt.m_PMode == 1)
			{
				m_FilePath = m_pGNSSOpt->m_ProjPath + FILEPATHSEP + "R_INS";
			}
			else if (m_SINSOpt.m_PMode == 2)
			{
				if (m_pGNSSOpt->m_PMode == GNSSMODE_SPP)
				{
					m_FilePath += "R_SPP_LCI";
				}
				else if (m_pGNSSOpt->m_PMode == GNSSMODE_PPP_STATIC || m_pGNSSOpt->m_PMode == GNSSMODE_PPP_KINEMA || m_pGNSSOpt->m_PMode == GNSSMODE_PPP_LEO)
				{
					m_FilePath += "R_PPP_LCI";
				}
				else if (m_pGNSSOpt->m_PMode == GNSSMODE_RTD || m_pGNSSOpt->m_PMode == GNSSMODE_RTK_KINEMA || m_pGNSSOpt->m_PMode == GNSSMODE_RTK_STATIC)
				{
					m_FilePath += "R_RTK_LCI";
				}
				else
				{
					if (m_SINSOpt.m_PMode > 1)
					{
						DumpException("GNSS process mode error!", LOGT_WARNING | LOGT_AFX, DUMP_LOCATION);
						return false;
					}
				}
			}
			else if (m_SINSOpt.m_PMode == 3)
			{
				if (m_pGNSSOpt->m_PMode == GNSSMODE_SPP)
				{
					m_FilePath += "R_SPP_TCI";
				}
				else if (m_pGNSSOpt->m_PMode == GNSSMODE_PPP_STATIC || m_pGNSSOpt->m_PMode == GNSSMODE_PPP_KINEMA || m_pGNSSOpt->m_PMode == GNSSMODE_PPP_LEO)
				{
					m_FilePath += "R_PPP_TCI";
				}
				else if (m_pGNSSOpt->m_PMode == GNSSMODE_RTD || m_pGNSSOpt->m_PMode == GNSSMODE_RTK_KINEMA || m_pGNSSOpt->m_PMode == GNSSMODE_RTK_STATIC)
				{
					m_FilePath += "R_RTK_TCI";
				}
				else
				{
					DumpException("GNSS process mode error!", LOGT_WARNING | LOGT_AFX, DUMP_LOCATION);
					return false;
				}
			}
			else if (m_SINSOpt.m_PMode == 4)
			{
				m_FilePath +=  "R_PAD";
			}
			else if (m_SINSOpt.m_PMode == 5)
			{
				if (m_pGNSSOpt->m_PMode == GNSSMODE_SPP)
				{
					m_FilePath += "R_SPP_STC";
				}
				else if (m_pGNSSOpt->m_PMode == GNSSMODE_PPP_STATIC || m_pGNSSOpt->m_PMode == GNSSMODE_PPP_KINEMA || m_pGNSSOpt->m_PMode == GNSSMODE_PPP_LEO)
				{
					m_FilePath += "R_PPP_STC";
				}
				else if (m_pGNSSOpt->m_PMode == GNSSMODE_RTD || m_pGNSSOpt->m_PMode == GNSSMODE_RTK_KINEMA || m_pGNSSOpt->m_PMode == GNSSMODE_RTK_STATIC)
				{
					m_FilePath += "R_RTK_STC";
				}
				else
				{
					DumpException("GNSS process mode error!", LOGT_WARNING | LOGT_AFX, DUMP_LOCATION);
					return false;
				}
			}
		}

		if (xmkdir(m_FilePath.c_str()) != 0 && xaccess(m_FilePath.c_str(), 0) < 0)
		{
			char msg[MAXSIZE] = { '\0' };
			sprintf(msg, "The file is not exist: %s%s.", gs_Placeholder.c_str(), m_FilePath.c_str());
			DumpException(string(msg), LOGT_WARNING | LOGT_AFX, DUMP_LOCATION);

			return false;
		}

		// LC File
		m_bflfFile = _BaseFileName(m_FilePath, FileTitle, gs_SINSFileTypes[SFT_BFLF]);
		m_brlfFile = _BaseFileName(m_FilePath, FileTitle, gs_SINSFileTypes[SFT_BRLF]);
		m_flfFile  = _BaseFileName(m_FilePath, FileTitle, gs_SINSFileTypes[SFT_FLF]);
		m_rlfFile  = _BaseFileName(m_FilePath, FileTitle, gs_SINSFileTypes[SFT_RLF]);
		m_bflsFile = _BaseFileName(m_FilePath, FileTitle, gs_SINSFileTypes[SFT_BFLS]);
		m_brlsFile = _BaseFileName(m_FilePath, FileTitle, gs_SINSFileTypes[SFT_BRLS]);
		m_flsFile  = _BaseFileName(m_FilePath, FileTitle, gs_SINSFileTypes[SFT_FLS]);
		m_rlsFile  = _BaseFileName(m_FilePath, FileTitle, gs_SINSFileTypes[SFT_RLS]);
		m_dlsFile  = _BaseFileName(m_FilePath, FileTitle, gs_SINSFileTypes[SFT_DLS]);
		m_dlfFile  = _BaseFileName(m_FilePath, FileTitle, gs_SINSFileTypes[SFT_DLF]);
		m_bdlsFile = _BaseFileName(m_FilePath, FileTitle, gs_SINSFileTypes[SFT_BDLS]);
		m_bdlfFile = _BaseFileName(m_FilePath, FileTitle, gs_SINSFileTypes[SFT_BDLF]);
		m_flfmsgFile = m_flfFile + "_msg";
		m_rlfmsgFile = m_rlfFile + "_msg";

		// TC File
		m_bftfFile = _BaseFileName(m_FilePath, FileTitle, gs_SINSFileTypes[SFT_BFTF]);
		m_brtfFile = _BaseFileName(m_FilePath, FileTitle, gs_SINSFileTypes[SFT_BRTF]);
		m_ftfFile  = _BaseFileName(m_FilePath, FileTitle, gs_SINSFileTypes[SFT_FTF]);
		m_rtfFile  = _BaseFileName(m_FilePath, FileTitle, gs_SINSFileTypes[SFT_RTF]);
		m_bftsFile = _BaseFileName(m_FilePath, FileTitle, gs_SINSFileTypes[SFT_BFTS]);
		m_brtsFile = _BaseFileName(m_FilePath, FileTitle, gs_SINSFileTypes[SFT_BRTS]);
		m_ftsFile  = _BaseFileName(m_FilePath, FileTitle, gs_SINSFileTypes[SFT_FTS]);
		m_rtsFile  = _BaseFileName(m_FilePath, FileTitle, gs_SINSFileTypes[SFT_RTS]);
		m_dtsFile  = _BaseFileName(m_FilePath, FileTitle, gs_SINSFileTypes[SFT_DTS]);
		m_dtfFile  = _BaseFileName(m_FilePath, FileTitle, gs_SINSFileTypes[SFT_DTF]);
		m_bdtsFile = _BaseFileName(m_FilePath, FileTitle, gs_SINSFileTypes[SFT_BDTS]);
		m_bdtfFile = _BaseFileName(m_FilePath, FileTitle, gs_SINSFileTypes[SFT_BDTF]);
		m_ftfmsgFile = m_ftfFile + "_msg";
		m_rtfmsgFile = m_rtfFile + "_msg";

		if (m_pGNSSOpt->m_bProjPath)
		{
			m_ExpFile = m_pGNSSOpt->m_ProjPath + FILEPATHSEP + "Export" + FILEPATHSEP + FileTitle + ".pos";
		}
		else
		{
			m_ExpFile = m_FilePath + FILEPATHSEP + FileTitle + ".pos";
		}

		return true;
	}


	bool CSINSApplication::ReadFiles()
	{
		M31EQU(m_SINSOpt.m_RotAngleIMU, m_IMUDataContainer.m_RotAngleIMU);

		if (m_SINSOpt.m_PMode == 1)
		{
			if (m_IMUDataContainer.open(m_IMUFile, m_SINSOpt.m_IMUBeginTime.GPSWeek) == false)
			{
				string msg("Fail to open IMU file.");
				DumpException(msg, LOGT_WARNING | LOGT_AFX, DUMP_LOCATION);
				return false;
			}
			return true;
		}
		
		// 组织生成ELC数据
		if (1)
		{
			if (AccessFile(m_GNSSFile) == false)
			{
				if (m_pGNSSOpt->m_PMode < GNSSMODE_SINGLE_MAX)
				{
					DumpException("GNSS-PPP solution file is not found.", LOGT_WARNING | LOGT_AFX);
				}
				else
				{
					DumpException("GNSS-RTK solution file is not found.", LOGT_WARNING | LOGT_AFX);
				}
				return false;
			}

			// 如果是DMR文件则进行转换
			bool bConvertDMI = false;
			string dstDMIFileName = "";
			if (m_DMIFile.rfind(".dmr") != string::npos)
			{
				dstDMIFileName = m_DMIFile; dstDMIFileName.replace(dstDMIFileName.rfind(".dmr"), 4, ".dmi");
				bConvertDMI = true;
			}
			else if (m_DMIFile.rfind(".DMR") != string::npos)
			{
				dstDMIFileName = m_DMIFile; dstDMIFileName.replace(dstDMIFileName.rfind(".DMR"), 4, ".dmi");
				bConvertDMI = true;
			}

			if (bConvertDMI)
			{
				CSINSDateConverter con;
				if (con.ConverteDMI(m_DMIFile, dstDMIFileName, m_SINSOpt.m_WheelSizeDMI)) m_DMIFile = dstDMIFileName;
				else m_DMIFile = "";
			}

			string iFiles[nELCFile];
			iFiles[0] = m_GNSSFile;
			iFiles[1] = m_POSFile;
			iFiles[2] = m_DMIFile;
			iFiles[3] = m_HEDFile;
			iFiles[4] = m_ZUPFile;
			iFiles[5] = m_ZIHFile;
			iFiles[6] = m_NHCFile;
			iFiles[7] = m_CUPFile;
			if (m_ELCDataContainer.ExternalELC(iFiles, &m_SINSOpt) == false)
			{
				string msg("Fail to init ELC data!");
				DumpException(msg, LOGT_ERROR | LOGT_AFX, DUMP_LOCATION);
				return false;
			}
			m_IMUGPSWeek = m_ELCDataContainer.m_GPSWeek;
		}

		if (m_SINSOpt.m_PMode == 2 || m_SINSOpt.m_PMode == 3 || m_SINSOpt.m_PMode == 5)
		{
			// 1. 判断文件个数
			if(m_pGNSSOpt->m_PMode <= GNSSMODE_SINGLE_MAX)
			{
				m_OBSDataContainers.resize(1);
			}
			else
			{
				// 多基站合并成一个基站文件进行处理
				if (m_pGNSSOpt->m_OBSFileNum > 2)
				{
					char cDir[1024] = { '\0' };    // 一定不要用m_GNSSOpt.m_GNSSFilePath
					getFileInfo(_C(m_SINSOpt.m_GNSSOpt.m_OBSFile[1].c_str()), NULL, NULL, NULL, cDir);
					string MultiBase = string(cDir) + FILEPATHSEP + "MultiBaseSite.o";
					string MultiBaseHead = string(cDir) + FILEPATHSEP + "MultiBaseSite.ob_HeadInf";

					m_bMultiBaseSite = true;
					GenerateMultiBaseOBSFile(m_OptFile);
					m_OBSFile[1] = m_SINSOpt.m_GNSSOpt.m_OBSFile[1] = MultiBase;
					ReadMultiBaseOBSHeadInf(MultiBaseHead, &m_SINSOpt.m_GNSSOpt);
					m_SINSOpt.m_GNSSOpt.m_OBSFileNum = 2;
				}

				m_OBSDataContainers.resize(m_pGNSSOpt->m_OBSFileNum);
			}

			// 2. 读取导航文件
			for (int i = 0; i<m_pGNSSOpt->m_NavFileMAX; i++)
			{
				// 单站解算模式
				if (m_pGNSSOpt->m_PMode <= GNSSMODE_SINGLE_MAX && i > 0) break;

				m_EphStores.m_SatSYS = m_pGNSSOpt->m_SatSYS;
				for (int j = 0; j <= NSYS; j++) m_EphStores.m_NavEphFile[j] = m_NavEphFile[j][i];
				
				// 只需流动站读取
				if (i == 0)
				{
					if (m_pGNSSOpt->m_PMode <= GNSSMODE_SINGLE_MAX && m_pGNSSOpt->m_ARMode > 0 && m_pGNSSOpt->m_ARType == 1) m_EphStores.m_FCBFile = m_FCBFile;

					if (m_pGNSSOpt->m_PMode <= GNSSMODE_SINGLE_MAX || m_pGNSSOpt->m_SPPEph == EPH_PRE || m_pGNSSOpt->m_RTKEph == EPH_PRE)
					{
						for (int j = 0; j < NSYS; j++)
						{
							m_EphStores.m_PreEphFile[j] = m_PreEphFile[j];
							m_EphStores.m_PreClkFile[j] = m_PreClkFile[j];
						}
					}
				}
				else
				{
					m_EphStores.m_FCBFile = "";
					for (int j = 0; j < NSYS; j++)
					{
						m_EphStores.m_PreEphFile[j] = "";
						m_EphStores.m_PreClkFile[j] = "";
					}
				}
				

				if (m_EphStores.openEphFile() == false)
				{
					DumpException("One navigation file is wrong.", LOGT_WARNING | LOGT_AFX, DUMP_LOCATION);
					continue;
				}
				else if (m_pGNSSOpt->m_PMode == GNSSMODE_PPP_STATIC || m_pGNSSOpt->m_PMode == GNSSMODE_PPP_KINEMA || m_pGNSSOpt->m_PMode == GNSSMODE_PPP_LEO)
				{
					bool bPreEph = false;
					bool bPreClk = false;

					for (int i = 0; i < NSYS; i++)
					{
						if (m_EphStores.m_bPreEph[i]) bPreEph = true;
						if (m_EphStores.m_bPreClk[i]) bPreClk = true;
					}

					if (bPreEph == false)
					{
						string msg("There is no precise ephemeris file!");
						DumpException(msg, LOGT_WARNING | LOGT_AFX, DUMP_LOCATION);
					}

					if (bPreClk == false)
					{
						string msg("There is no precise clock file!");
						DumpException(msg, LOGT_WARNING | LOGT_AFX, DUMP_LOCATION);
					}
				}

				bool bbrdc = false;
				for (int j = 0; j < NSYS; j++)
				{
					if ((m_pGNSSOpt->m_SatSYS&Index2Sys(j)) && m_EphStores.m_bNavEph[j]) { bbrdc = true; break; }
				}

				if (bbrdc == false && m_pGNSSOpt->m_SPPEph == 0)
				{
					string msg("There are no matched navigation files.");
					DumpException(msg, LOGT_ERROR | LOGT_AFX, DUMP_LOCATION);
					return false; // 只输出警告,不返回
				}
			}

			// 3. 读取观测文件
			for (int i = 0; i < m_pGNSSOpt->m_OBSFileNum; i++)
			{
				// 单站解算模式
				if (m_pGNSSOpt->m_PMode <= GNSSMODE_SINGLE_MAX && i > 0) break;

				string ObsFile = m_OBSFile[i] + "b";
				bool bConvert = false;
                if (xaccess(ObsFile.c_str(), 0) == -1)
				{
					bConvert = true;
				}
				else
				{
					char bValid[13] = { '\0' };
					FILE* fp = fopen(ObsFile.c_str(), "rb");
					if (!fp) return false;
					fread(&bValid, sizeof(char), strlen(OBSVALIDFIELD), fp); // 检查第一个字段,验证转换是否成功
					fclose(fp);
					if (strcmp(bValid, OBSVALIDFIELD) != 0)
					{
						bConvert = true;
						DumpException("Fail to validate conversion, convert again.", LOGT_INFO, DUMP_LOCATION);
					}
				}

			#ifndef IPS_BP
				if (bConvert && ConvertGNSSObsFile(m_OBSFile[i], ObsFile) == false)
				{
					string msg("Fail to convert the observation file!");
					DumpException(msg, LOGT_ERROR | LOGT_AFX, DUMP_LOCATION);
					return false;
				}
			#endif

				m_OBSFile[i] = ObsFile;

				// 如果是VRS基站文件,需要如下操作
				if (i == 1 && m_SINSOpt.m_GNSSOpt.m_BaseXYZVRSFile != "" && m_SINSOpt.m_GNSSOpt.m_VRSXYZList.size() > 1)
				{
					m_bMultiBaseSite = true;
					m_OBSDataContainers[i].m_VRSXYZList = m_SINSOpt.m_GNSSOpt.m_VRSXYZList;
				}

				// Obs File
				if (m_OBSDataContainers[i].open(m_OBSFile[i], m_SINSOpt.m_GNSSOpt.m_ProcRate, i > 0 ? false : true) == false)
				{
					string msg("Fail to open the observation file!");
					DumpException(msg, LOGT_ERROR | LOGT_AFX, DUMP_LOCATION);
					return false;
				}

				if (!(i > 0 && m_bMultiBaseSite))
				{
					// 天线高
					if (m_pGNSSOpt->m_bAntDelFromOBS)
					{
						if (!(gs_bHCDH || gs_bLJYY)) M31EQU(m_OBSDataContainers[i].m_FFSHead.ant, m_pGNSSOpt->m_AntDel[i]);
					}

					// 天线类型
					m_pGNSSOpt->m_AntType[i] = string(m_OBSDataContainers[i].m_FFSHead.antType);
				}

				char filetitle[MAXSIZE] = {'\0'};
				if (getFileInfo(m_OBSFile[i].c_str(), filetitle) == false)
				{
					string msg("Fail to get the observation file information!");
					DumpException(msg, LOGT_WARNING | LOGT_AFX, DUMP_LOCATION);
					return false;
				}
				m_pGNSSOpt->m_siteName[i] = string(filetitle);

				// 使用导出多普勒替代原始多普勒观测值
				if (m_pGNSSOpt->m_bDerivedDoppler)
				{
					m_pGNSSOpt->m_SampleTime = m_OBSDataContainers[i].m_FFSHead.dt; // 内部GDD需要
					m_OBSDataContainers[i].DeriveDoppler(m_pGNSSOpt, &m_EphStores);
				}
			}

			// 重新赋值,得到数据采样时间,这里就以流动站为准
			m_pGNSSOpt->m_SampleTime = m_OBSDataContainers[0].m_FFSHead.dt;
			if (m_pGNSSOpt->m_bGNSSProcAll)
			{
				m_pGNSSOpt->m_BeginTime = m_OBSDataContainers[0].m_BeginTime;
				m_pGNSSOpt->m_EndTime = m_OBSDataContainers[0].m_EndTime;
			}
			
			// 更新AntMode信息
			m_pGNSSOpt->UpdateAntMode();

			// IGS File
			m_pGNSSOpt->getGNSSFile(m_OBSDataContainers[0].m_BeginTime);
			m_IGSProducts.openDCB(m_pGNSSOpt->m_DCBFile[0]);
			m_IGSProducts.openDCB(m_pGNSSOpt->m_DCBFile[1]);
			m_IGSProducts.openDCB(m_pGNSSOpt->m_DCBFile[2]);
			m_IGSProducts.openPCV(m_pGNSSOpt->m_ATXFile);
			m_IGSProducts.openERP(m_pGNSSOpt->m_ERPFile);
			m_IGSProducts.openBLQ(m_pGNSSOpt->m_BLQFile);
			m_IGSProducts.StartIGSProducts(m_pGNSSOpt);

			// 如果存在跨周,这边会有危险
			m_IMUGPSWeek = m_OBSDataContainers[0].m_BeginTime.GPSWeek;
		}

		if (m_IMUGPSWeek <= 0)
		{
			string msg("GPS Week of IMU data is wrong.");
			DumpException(msg, LOGT_WARNING | LOGT_AFX, DUMP_LOCATION);
			return false;
		}

		// IMU File
		m_SINSOpt.m_IMUBeginTime.GPSWeek = m_IMUGPSWeek;
		m_SINSOpt.m_IMUEndTime.GPSWeek   = m_IMUGPSWeek;
		if (_TOW(m_SINSOpt.m_IMUBeginTime) > _TOW(m_SINSOpt.m_IMUEndTime)) m_SINSOpt.m_IMUEndTime.GPSWeek++;


		SelectZeroVelInitTime(&m_SINSOpt, &m_OBSDataContainers[0], &m_IMUDataContainer, &m_EphStores, &m_IGSProducts);
		string ZUPTFile = m_IMUFile + "_zpt";
		if (m_IMUDataContainer.open(m_IMUFile, m_IMUGPSWeek, ZUPTFile) == false)
		{
			string msg("Fail to open IMU file.");
			DumpException(msg, LOGT_WARNING | LOGT_AFX, DUMP_LOCATION);
			return false;
		}
		m_SINSOpt.m_IMUSample = m_IMUDataContainer.m_FFSHead.dt;

		// 生成最终的ELC
		if (m_ELCDataContainer.InternalELC(&m_IMUDataContainer) == false || m_ELCDataContainer.getFinalELC() == false)
		{
			string msg("Fail to init ELC data!");
			DumpException(msg, LOGT_ERROR | LOGT_AFX, DUMP_LOCATION);
			return false;
		}

		return true;
	}


	void CSINSApplication::AssignSINSHead(CSINSSolutionHead *m_SINSHead)
	{
		// 将SINSApplication中的大量信息传入到头信息中
		m_SINSHead->m_OptFile = m_OptFile;
		m_SINSHead->m_FilePath = m_FilePath;

		m_SINSHead->m_bGNSSProcAll = m_SINSOpt.m_GNSSOpt.m_bGNSSProcAll;
		m_SINSHead->m_GNSS_PMode = m_SINSOpt.m_GNSSOpt.m_PMode;
		m_SINSHead->m_SatSYS = m_SINSOpt.m_GNSSOpt.m_SatSYS;
		m_SINSHead->m_GNSS_FilterType = m_SINSOpt.m_GNSSOpt.m_FilterType;
		m_SINSHead->m_BaselineType = m_SINSOpt.m_GNSSOpt.m_RTKType;
		m_SINSHead->m_ATXFile = m_SINSOpt.m_GNSSOpt.m_ATXFile;
		m_SINSHead->m_ProcRate = m_SINSOpt.m_GNSSOpt.m_ProcRate;
		if(m_OBSDataContainers.size() > 0) m_SINSHead->m_OBSDataRate = m_OBSDataContainers[0].m_FFSHead.dt;
		m_SINSHead->m_ElevMask = m_SINSOpt.m_GNSSOpt.m_ElevMask;
		if (m_OBSDataContainers.size() > 0) m_SINSHead->m_RecordNum = m_OBSDataContainers[0].m_RecordNum;
		m_SINSHead->m_BaseSiteNum = m_SINSOpt.m_GNSSOpt.m_BaseSiteNum;

		for (int i = 0; i < 3; i++) m_SINSHead->m_DCBFile[i] = m_SINSOpt.m_GNSSOpt.m_DCBFile[i];

		for (int i = 0; i < NSYS; i++)
		{
			m_SINSHead->m_ErrorPLR[i] = m_SINSOpt.m_GNSSOpt.m_ErrorPLR[i];
			m_SINSHead->m_Error_EF[i] = m_SINSOpt.m_GNSSOpt.m_Error_EF[i];
			m_SINSHead->m_PreEphFile[i] = m_PreEphFile[i];
			m_SINSHead->m_PreClkFile[i] = m_PreClkFile[i];
		}
		
		for (int i = 0; i < m_OBSDataContainers.size(); i++)
		{
			m_SINSHead->m_RecType[i] = m_OBSDataContainers[i].m_FFSHead.recType;
			m_SINSHead->m_AntType[i] = m_OBSDataContainers[i].m_FFSHead.antType;
		}

		for (int i = 0; i < m_SINSOpt.m_GNSSOpt.m_BaseSiteNum + 1; i++)
		{
			m_SINSHead->m_OBSFile[i] = m_OBSFile[i];
			m_SINSHead->m_AntMode[i] = m_SINSOpt.m_GNSSOpt.m_AntMode[i];

			for (int j = 0; j < 3; j++)
			{
				m_SINSHead->m_BaseXYZ[i][j] = m_SINSOpt.m_GNSSOpt.m_BaseXYZ[i][j];
				m_SINSHead->m_AntDel[i][j] = m_SINSOpt.m_GNSSOpt.m_AntDel[i][j];
				m_SINSHead->m_AntOffset[i][j] = m_SINSOpt.m_GNSSOpt.m_AntOffset[i][j];
			}
		}

		for (int i = 0; i < NSYS + 1; i++)
		{
			for (int j = 0; j < m_SINSOpt.m_GNSSOpt.m_BaseSiteNum + 1; j++)
			{
				m_SINSHead->m_NavEphFile[i][j] = m_NavEphFile[i][j];
			}
		}

		if (m_OBSDataContainers.size() > 0) m_SINSHead->m_OBSDataBeginTime = m_OBSDataContainers[0].m_BeginTime;
		if (m_OBSDataContainers.size() > 0) m_SINSHead->m_OBSDataEndTime = m_OBSDataContainers[0].m_EndTime;
		m_SINSHead->m_ProcBeginTime = m_SINSOpt.m_GNSSOpt.m_BeginTime;
		m_SINSHead->m_ProcEndTime = m_SINSOpt.m_GNSSOpt.m_EndTime;

		m_SINSHead->m_yTime = getCurrentTime();

		m_SINSHead->m_GSP = m_SINSOpt.m_GNSSOpt.m_GSP;

		m_SINSHead->m_IMUFile = m_IMUFile;
		m_SINSHead->m_GNSSFile = m_GNSSFile;
		m_SINSHead->m_POSFile = m_POSFile;
		m_SINSHead->m_DMIFile = m_DMIFile;
		m_SINSHead->m_HEDFile = m_HEDFile;
		m_SINSHead->m_ZUPFile = m_ZUPFile;

		m_SINSHead->m_SINS_PMode = m_SINSOpt.m_PMode;
		m_SINSHead->m_SINS_FilterType = m_SINSOpt.m_FilterType;
		m_SINSHead->m_SSP = m_SINSOpt.m_SSP;
		m_SINSHead->m_bIMUProcAll = m_SINSOpt.m_bIMUProcAll;
		m_SINSHead->m_IMUProcBeginTime = m_SINSOpt.m_IMUBeginTime;
		m_SINSHead->m_IMUProcEndTime = m_SINSOpt.m_IMUEndTime;

		m_SINSHead->m_IMUName = m_IMUDataContainer.m_FFSHead.szImuName;
		m_SINSHead->m_IMUDataBeginTime = m_IMUDataContainer.m_BeginTime;
		m_SINSHead->m_IMUDataEndTime = m_IMUDataContainer.m_EndTime;
		m_SINSHead->m_IMUDataRate = 1.0 / m_IMUDataContainer.m_FFSHead.dDataRateHz;
		M31EQU(m_SINSOpt.m_LeverArmbGNSS0, m_SINSHead->m_LeverArmbGNSS);
		M31EQU(m_SINSOpt.m_RotAngleIMU, m_SINSHead->m_RotAngleIMU);

		//该项为true才会进行写入头文件这一操作
		m_SINSHead->m_bAssignSINSHead = true;
	}


	bool CSINSApplication::runForwardLC()
	{
		ZeroStruct(m_SINSSolRpt[0], SOLRPT);
		m_AppInfo[0].bFinish = false;
		m_AppInfo[0].bValid = true;
		strcpy(m_AppInfo[0].Title, "Forward KF: LCI");
		if (gs_bUsedPThread) { DUMP_OPEN_Forward(m_flfmsgFile) }
		else { DUMP_OPEN(m_flfmsgFile) }

		bool bReverse = false;
		bool bInitELC = false;
		CSINSOption SINSOpt = m_SINSOpt;
		if (SINSOpt.m_bIMUProcAll)
		{
			SINSOpt.m_IMUBeginTime = m_IMUDataContainer.m_BeginTime;
			SINSOpt.m_IMUEndTime = m_IMUDataContainer.m_EndTime;
		}

		if (m_ELCDataContainer.m_bValid)
		{
			SINSOpt.m_ELCBeginTime = m_ELCDataContainer.m_BeginTime;
			SINSOpt.m_ELCEndTime = m_ELCDataContainer.m_EndTime;
			bInitELC = true;
		}

		// 只有ELC可用时进行SINS初始化对准,获取初始位置速度和姿态
		if (bInitELC && m_SINSInitState.init(bReverse, &SINSOpt, &m_ELCDataContainer, &m_IMUDataContainer))
		{
			if(!m_SINSInitState.getSINSInitState()) return false;
		}
		else
		{
			string msg("There is no ELC file, fail to get SINS init state!");
			DumpException(msg, LOGT_WARNING | LOGT_AFX, DUMP_LOCATION);
			return false;
		}

		GPSTIME gt, IMUTime;

		// 1. 读取SINS数据
		CIMUFFStream IMUFFStream;
		IMUFFStream.m_IMUDataGapTol = SINSOpt.m_IMUDataGapTol;
		IMUFFStream.setDataContainer(&m_IMUDataContainer);
		if (IMUFFStream.init(bReverse) == false) return false;
		IMUFFStream.getOneRecordData();
		IMUTime = IMUFFStream.m_time;
		while (IMUFFStream.peakOneRecordTime() <= SINSOpt.m_IMUBeginTime) IMUFFStream.getOneRecordData();
		if (IMUFFStream.CheckDataIsEOF()) return false;


		// 2. 读取ELC数据
		CELCFFStream ELCFFStream;
		ELCFFStream.setDataContainer(&m_ELCDataContainer);
		if (ELCFFStream.init(bReverse) == false) return false;
		ELCFFStream.getOneRecordData();
		gt = ELCFFStream.m_time;
		while (ELCFFStream.peakOneRecordTime() <= SINSOpt.m_IMUBeginTime) ELCFFStream.getOneRecordData();
		if (ELCFFStream.CheckDataIsEOF()) return false;


		// 4. 初始化LC
		if (1) // 括号里的类运行完后会自动调用析构函数,关闭相应文件
		{
			CSINSSolution SINSSol;
			SINSSol.m_pSINSOpt = &SINSOpt;
			AssignSINSHead(&SINSSol.m_SINSHead);
			if (SINSSol.init(m_flfFile, true, false) == false) return false;

			CLooseCouplingBase* pForwardLC;
			if (SINSOpt.m_NaviSys == NAVI_E) pForwardLC = new CLooseCouplingECEF();
			else                             pForwardLC = new CLooseCouplingLocal();
			pForwardLC->m_pSINSSol     = &SINSSol;
			pForwardLC->m_pIMUFFStream = &IMUFFStream;
			pForwardLC->m_pELCFFStream = &ELCFFStream;
			pForwardLC->m_pSINSOpt     = &SINSOpt;
			if (pForwardLC->initLC(bReverse) == false) return false;


#ifdef _TwiceProc_LC_ON_

			CIMUFFStream IMUFFStreamTmp = IMUFFStream;
			CSINSSolution SINSSolTmp;
			CLooseCouplingBase* pForwardLCTmp = NULL;
			
			if (ls_bCheckScene)
			{
				if (SINSOpt.m_NaviSys == NAVI_E) pForwardLCTmp = new CLooseCouplingECEF();
				else                             pForwardLCTmp = new CLooseCouplingLocal();
				pForwardLCTmp->m_pSINSSol =     &SINSSolTmp;
				pForwardLCTmp->m_pIMUFFStream = &IMUFFStreamTmp;
				pForwardLCTmp->m_pELCFFStream = &ELCFFStream;
				pForwardLCTmp->m_pSINSOpt     = &SINSOpt;
				if (pForwardLCTmp->initLC(bReverse) == false) return false;
			}

#endif // _TwiceProc_LC_ON_


			// 进度条
			double data_len = fabs(SINSOpt.m_IMUEndTime - SINSOpt.m_IMUBeginTime);
			double data_i = 0;
			double fProgress = 0.0;
			int iProgress = 0, iProgressPre = 0;
			bool flag_ELC = true;
			bool flag_IMU = true;
			bool bELCData = true;
			bool bIMUData = true;
			int  errorID = 0;

			while (1)
			{
				if (flag_ELC)
				{
					if (ELCFFStream.getOneRecordData() == false) break;
					gt = ELCFFStream.m_time;
				}

				if (flag_IMU)
				{
					if (IMUFFStream.getOneRecordData(true) == false) break;
					IMUTime = IMUFFStream.m_time;

#ifdef _TwiceProc_LC_ON_
					if (ls_bCheckScene)
					{
						if (IMUFFStreamTmp.getOneRecordData(true) == false) break;
					}
#endif					
				}

				if (IMUTime > SINSOpt.m_IMUEndTime) break;

				flag_ELC = flag_IMU = bELCData = bIMUData = false;

				// 1. 判断LC的操作
				if (IMUTime - IMUFFStream.m_DeltT < gt && gt <= IMUTime)
				{
					// 执行观测更新,读取两者数据
					bELCData = true;
					flag_ELC = true;
					flag_IMU = true;
				}
				else if (gt > IMUTime)
				{
					// 执行状态更新,读取IMU数据
					bIMUData = true;
					flag_IMU = true;
				}
				else
				{
					// 不进行任何更新,读取ELC数据
					flag_ELC = true;
				}

				// lj:毕业设计
				// 在这里读取IMU数据吗


				if (bELCData)
				{
					if (gt != IMUTime) IMUFFStream.setMeasUpdate(gt); // 观测更新时设置下进行内插
#ifdef _TwiceProc_LC_ON_
					if (ls_bCheckScene)
					{
						if (gt != IMUTime) IMUFFStreamTmp.setMeasUpdate(gt); // 观测更新时设置下进行内插

						int Quality = ELCFFStream.m_ELCData.GNSS_DATA.Quality; // 记录原本的Q

						pForwardLCTmp->runOneLC(gt, true);
						if ((fabs(SINSSolTmp.m_MisclosurePos[0]) > 0.2 || fabs(SINSSolTmp.m_MisclosurePos[1]) > 0.2 || fabs(SINSSolTmp.m_MisclosurePos[2]) > 0.2) && (SINSSolTmp.m_IMUFlag&ELCTYPE_POS))
						{
							Quality = 7;
							ls_nCheckPass = 0;
						}
						else if (SINSSolTmp.m_IMUFlag&ELCTYPE_POS)
						{
							ls_nCheckPass++;
						}
						else
						{
							Quality = 7;
						}

						// 检测没问题5个历元内仍不予通过
						if (SINSOpt.m_bTwiceProcLC_NumCheck && ls_nCheckPass < 5) Quality = 7;

						// 满足下列条件认为GNSS没问题，予以通过
						if ((ELCFFStream.m_ELCData.GNSS_DATA.ARRatio > 6 && ELCFFStream.m_ELCData.GNSS_DATA.nSat_ARHold > 6 && ELCFFStream.m_ELCData.GNSS_DATA.AmbState >= ARSTATE_HOLD && MatrixNorm2(3, 1, ELCFFStream.m_ELCData.GNSS_DATA.SepENU) < 0.5) ||
							(ELCFFStream.m_ELCData.GNSS_DATA.ARRatio > 10 && ELCFFStream.m_ELCData.GNSS_DATA.PDOP < 2.0 && MatrixNorm2(3, 1, ELCFFStream.m_ELCData.GNSS_DATA.SepENU) < 0.5))
							Quality = ELCFFStream.m_ELCData.GNSS_DATA.Quality;

						ELCFFStream.m_ELCData.GNSS_DATA.Quality = Quality;
					}

#endif
					errorID = pForwardLC->runOneLC(gt, true);
				}
				else if (bIMUData)
				{
#ifdef _TwiceProc_LC_ON_
					if (ls_bCheckScene) pForwardLCTmp->runOneLC(IMUTime, false);
#endif
					errorID = pForwardLC->runOneLC(IMUTime, false);
				}

				// 观测更新成功,输出有用结果
				if (errorID == 0 && bELCData)
				{
					if (gs_bIPSAPP || gs_bHCDH || gs_bLJYY) // APP界面信息
					{
						data_i = fabs(SINSSol.m_time - SINSOpt.m_IMUBeginTime);
						if (data_i < data_len)
						{
							fProgress = (data_i / data_len) * 100.0; // 进度条
							iProgress = ceil(fProgress);
						}
						SINSSol.FlushAppInfo(fProgress, &m_AppInfo[0], m_SINSSolRpt);
						if (iProgress != iProgressPre)
						{
							DumpProgress(iProgress, m_AppInfo[0]);
						}
						iProgressPre = iProgress;
					}
				}
				else if (errorID == 2) break;
			}

			// 删除,防止内存泄露,不删除将报错
			delete pForwardLC;

#ifdef _TwiceProc_LC_ON_
			if (pForwardLCTmp) delete pForwardLCTmp;
#endif
			// 作为下次后向滤波的初始对准结果
			m_SINSInitState.getNxtSINSInitState(SINSSol.m_time, SINSSol.m_XYZPos, SINSSol.m_XYZVel, SINSSol.m_Att, SINSSol.m_GyrosBias, SINSSol.m_AccelBias);
		}

		if (gs_bUsedPThread) { DUMP_CLOSE_Forward }
		else { DUMP_CLOSE }

		if (SINSOpt.m_bRTS)
		{
			strcpy(m_AppInfo[0].Title, "Forward RTS: LCI");
			SINS_RTS_Smoother(false, m_OptFile, m_bflfFile, m_flsFile, &m_SINSOpt, &m_AppInfo[0].progress);
		}

		m_AppInfo[0].bFinish = true;
		m_AppInfo[0].bValid = false;

		return true;
	}


	bool CSINSApplication::runReverseLC()
	{
		ZeroStruct(m_SINSSolRpt[1], SOLRPT);
		m_AppInfo[1].bFinish = false;
		m_AppInfo[1].bValid = true;
		strcpy(m_AppInfo[1].Title, "Reverse KF: LCI");
		if (gs_bUsedPThread) { DUMP_OPEN_Reverse(m_rlfmsgFile) }
		else { DUMP_OPEN(m_rlfmsgFile) }

		bool bReverse = true;
		bool bInitELC = false;
		CSINSOption SINSOpt = m_SINSOpt;
		if (SINSOpt.m_bIMUProcAll)
		{
			SINSOpt.m_IMUBeginTime = m_IMUDataContainer.m_BeginTime;
			SINSOpt.m_IMUEndTime = m_IMUDataContainer.m_EndTime;
		}

		if (m_ELCDataContainer.m_bValid)
		{
			SINSOpt.m_ELCBeginTime = m_ELCDataContainer.m_BeginTime;
			SINSOpt.m_ELCEndTime = m_ELCDataContainer.m_EndTime;
			bInitELC = true;
		}

		// 只有ELC可用时进行SINS初始化对准,获取初始位置速度和姿态
		if (bInitELC && m_SINSInitState.init(bReverse, &SINSOpt, &m_ELCDataContainer, &m_IMUDataContainer))
		{
			if (!m_SINSInitState.getSINSInitState()) return false;
		}
		else
		{
			string msg("There is no ELC file, fail to get SINS init state!");
			DumpException(msg, LOGT_WARNING | LOGT_AFX, DUMP_LOCATION);
			return false;
		}

		GPSTIME gt, IMUTime;

		// 2. 读取SINS数据
		CIMUFFStream IMUFFStream;
		IMUFFStream.m_IMUDataGapTol = SINSOpt.m_IMUDataGapTol;
		IMUFFStream.setDataContainer(&m_IMUDataContainer);
		if (IMUFFStream.init(bReverse) == false) return false;
		IMUFFStream.getOneRecordData();
		IMUTime = IMUFFStream.m_time;
		while (IMUFFStream.peakOneRecordTime() >= SINSOpt.m_IMUEndTime) IMUFFStream.getOneRecordData();
		if (IMUFFStream.CheckDataIsEOF()) return false;

		// 只是为了与原来的对齐,可以去掉???
		// IMUFFStream.getOneRecordData();

		// 3. 读取ELC数据
		CELCFFStream ELCFFStream;
		ELCFFStream.setDataContainer(&m_ELCDataContainer);
		if (ELCFFStream.init(bReverse) == false) return false;
		ELCFFStream.getOneRecordData();
		gt = ELCFFStream.m_time;
		while (ELCFFStream.peakOneRecordTime() >= SINSOpt.m_IMUEndTime) ELCFFStream.getOneRecordData();
		if (ELCFFStream.CheckDataIsEOF()) return false;


		// 4. 初始化LC
		if (1) // 括号里的类运行完后会自动调用析构函数,关闭相应文件
		{
			CSINSSolution SINSSol;
			SINSSol.m_pSINSOpt = &SINSOpt;
			AssignSINSHead(&SINSSol.m_SINSHead);
			if (SINSSol.init(m_rlfFile, true, false) == false) return false;

			CLooseCouplingBase* pReverseLC;
			if (SINSOpt.m_NaviSys == NAVI_E) pReverseLC = new CLooseCouplingECEF();
			else                             pReverseLC = new CLooseCouplingLocal();
			pReverseLC->m_pSINSSol = &SINSSol;
			pReverseLC->m_pIMUFFStream = &IMUFFStream;
			pReverseLC->m_pELCFFStream = &ELCFFStream;
			pReverseLC->m_pSINSOpt = &SINSOpt;
			if (pReverseLC->initLC(bReverse) == false) return false;

#ifdef _TwiceProc_LC_ON_

			CIMUFFStream IMUFFStreamTmp = IMUFFStream;
			CSINSSolution SINSSolTmp;
			CLooseCouplingBase* pReverseLCTmp = NULL;
			
			if (ls_bCheckScene)
			{
				if (SINSOpt.m_NaviSys == NAVI_E) pReverseLCTmp = new CLooseCouplingECEF();
				else                             pReverseLCTmp = new CLooseCouplingLocal();
				pReverseLCTmp->m_pSINSSol =     &SINSSolTmp;
				pReverseLCTmp->m_pIMUFFStream = &IMUFFStreamTmp;
				pReverseLCTmp->m_pELCFFStream = &ELCFFStream;
				pReverseLCTmp->m_pSINSOpt =     &SINSOpt;
				if (pReverseLCTmp->initLC(bReverse) == false) return false;
			}

#endif // _TwiceProc_LC_ON_

			// 进度条
			double data_len = fabs(SINSOpt.m_IMUEndTime - SINSOpt.m_IMUBeginTime);
			double data_i = data_len;
			double fProgress = 0.0;
			int iProgress = 0, iProgressPre = 0;
			bool flag_ELC = true;
			bool flag_IMU = true;
			bool bELCData = true;
			bool bIMUData = true;
			int  errorID = 0;

			while (1)
			{
				if (flag_ELC)
				{
					if (ELCFFStream.getOneRecordData() == false) break;
					gt = ELCFFStream.m_time;
				}

				if (flag_IMU)
				{
					if (IMUFFStream.getOneRecordData(true) == false) break;
					IMUTime = IMUFFStream.m_time;

#ifdef _TwiceProc_LC_ON_
					if (ls_bCheckScene)
					{
						if (IMUFFStreamTmp.getOneRecordData(true) == false) break;
					}
#endif
				}

				if (IMUTime < SINSOpt.m_IMUBeginTime) break;

				flag_ELC = flag_IMU = bELCData = bIMUData = false;

				// 1. 判断LC的操作
				if (IMUTime + IMUFFStream.m_DeltT > gt && gt >= IMUTime)
				{
					// 执行观测更新,读取两者数据
					bELCData = true;
					flag_ELC = true;
					flag_IMU = true;
				}
				else if (gt < IMUTime)
				{
					// 执行状态更新,读取IMU数据
					bIMUData = true;
					flag_IMU = true;
				}
				else
				{
					// 不进行任何更新,读取ELC数据
					flag_ELC = true;
				}

				if (bELCData)
				{
					if (gt != IMUTime) IMUFFStream.setMeasUpdate(gt); // 观测更新时设置下进行内插
#ifdef _TwiceProc_LC_ON_
					if (ls_bCheckScene)
					{
						if (gt != IMUTime) IMUFFStreamTmp.setMeasUpdate(gt); // 观测更新时设置下进行内插

						int Quality = ELCFFStream.m_ELCData.GNSS_DATA.Quality; // 记录原本的Q

						pReverseLCTmp->runOneLC(gt, true);
						if ((fabs(SINSSolTmp.m_MisclosurePos[0]) > 0.2 || fabs(SINSSolTmp.m_MisclosurePos[1]) > 0.2 || fabs(SINSSolTmp.m_MisclosurePos[2]) > 0.2) && (SINSSolTmp.m_IMUFlag&ELCTYPE_POS))
						{
							Quality = 7;
							ls_nCheckPass = 0;
						}
						else if (SINSSolTmp.m_IMUFlag&ELCTYPE_POS)
						{
							ls_nCheckPass++;
						}
						else
						{
							Quality = 7;
						}

						// 检测没问题5个历元内仍不予通过
						if (SINSOpt.m_bTwiceProcLC_NumCheck && ls_nCheckPass < 5) Quality = 7;

						// 满足下列条件认为GNSS没问题，予以通过
						if ((ELCFFStream.m_ELCData.GNSS_DATA.ARRatio > 6 && ELCFFStream.m_ELCData.GNSS_DATA.nSat_ARHold > 6 && ELCFFStream.m_ELCData.GNSS_DATA.AmbState >= ARSTATE_HOLD && MatrixNorm2(3, 1, ELCFFStream.m_ELCData.GNSS_DATA.SepENU) < 0.5) ||
							(ELCFFStream.m_ELCData.GNSS_DATA.ARRatio > 10 && ELCFFStream.m_ELCData.GNSS_DATA.PDOP < 2.0 && MatrixNorm2(3, 1, ELCFFStream.m_ELCData.GNSS_DATA.SepENU) < 0.5))
							Quality = ELCFFStream.m_ELCData.GNSS_DATA.Quality;

						ELCFFStream.m_ELCData.GNSS_DATA.Quality = Quality;
					}
					
#endif
					errorID = pReverseLC->runOneLC(gt, true);
				}
				else if (bIMUData)
				{
#ifdef _TwiceProc_LC_ON_
					if (ls_bCheckScene) pReverseLCTmp->runOneLC(IMUTime, false);
#endif
					errorID = pReverseLC->runOneLC(IMUTime, false);
				}

				// 观测更新成功,输出有用结果
				if (errorID == 0 && bELCData)
				{
					if (gs_bIPSAPP || gs_bHCDH || gs_bLJYY) // APP界面信息
					{
						if (gs_bHCDH || gs_bLJYY) data_i = fabs(SINSSol.m_time - SINSOpt.m_IMUEndTime);
						else if (gs_bIPSAPP) data_i = fabs(SINSSol.m_time - SINSOpt.m_IMUBeginTime);
						if (data_i < data_len)
						{
							fProgress = (data_i / data_len) * 100.0; // 进度条
							iProgress = ceil(fProgress);
						}
						SINSSol.FlushAppInfo(fProgress, &m_AppInfo[1], m_SINSSolRpt + 1);
						if (iProgress != iProgressPre)
						{
							DumpProgress(iProgress, m_AppInfo[1]);
						}
						iProgressPre = iProgress;
					}
				}
				else if (errorID == 2) break;
			}

			// 删除,防止内存泄露,不删除将报错
			delete pReverseLC;

#ifdef _TwiceProc_LC_ON_
			if (pReverseLCTmp) delete pReverseLCTmp;
#endif
			// 作为下次前向滤波的初始对准结果
			m_SINSInitState.getNxtSINSInitState(SINSSol.m_time, SINSSol.m_XYZPos, SINSSol.m_XYZVel, SINSSol.m_Att, SINSSol.m_GyrosBias, SINSSol.m_AccelBias);
		}

		if (gs_bUsedPThread) { DUMP_CLOSE_Reverse }
		else { DUMP_CLOSE }

		ReverseFile(m_rlfFile, 2);

		if (SINSOpt.m_bRTS)
		{
			strcpy(m_AppInfo[1].Title, "Reverse RTS: LCI");
			SINS_RTS_Smoother(true, m_OptFile, m_brlfFile, m_rlsFile, &m_SINSOpt, &m_AppInfo[1].progress);
		}

		m_AppInfo[1].bFinish = true;
		m_AppInfo[1].bValid = false;

		return true;
	}


	bool CSINSApplication::runLC()
	{
//#ifdef _TwiceProc_LC_ON_
		string filename = m_pGNSSOpt->m_ProjPath + FILEPATHSEP + m_pGNSSOpt->m_ProjName + ".rpt";
		ls_bCheckScene = CheckScene(filename);
		if (gs_bHCDH || gs_bLJYY) ls_bCheckScene = true;

		// 如果惯导是低成本的，不要开启
		if (ls_bCheckScene && (m_SINSOpt.m_SSP.PNSD_AccelBiasX > 1e-5 || m_SINSOpt.m_SSP.PNSD_AccelBiasY > 1e-5 || m_SINSOpt.m_SSP.PNSD_AccelBiasZ > 1e-5)
			&& (m_SINSOpt.m_SSP.PNSD_GyrosBiasX > 1e-5 * D2R || m_SINSOpt.m_SSP.PNSD_GyrosBiasY > 1e-5 * D2R || m_SINSOpt.m_SSP.PNSD_GyrosBiasZ > 1e-5 * D2R))
			ls_bCheckScene = false;

//#endif // _TwiceProc_LC_ON_

		if(m_SINSOpt.m_FilterType == 0 || m_SINSOpt.m_FilterType == 2)
		{
			m_AppInfo[0].FilterType = 0;
			if (gs_bUsedPThread)
			{
				gs_Forward_Thread = new thread(&sinstk::CSINSApplication::runForwardLC, this);
			}
			else
			{
				runForwardLC();
			}
		}

		if (gs_bHCDH || gs_bLJYY)
		{
			if (m_SINSInitDMIModel.init(&m_SINSOpt, m_bflsFile, m_DMIFile)) m_SINSInitDMIModel.runInstallEstimate();
		}

		//XZ估计安置误差
		//if (m_SINSInitDMIModel.init(&m_SINSOpt, m_bflfFile, "")) m_SINSInitDMIModel.runInstallEstimate();
		
		if(m_SINSOpt.m_FilterType == 1 || m_SINSOpt.m_FilterType == 2)
		{
			m_AppInfo[1].FilterType = 1;
			if (gs_bUsedPThread)
			{
				gs_Reverse_Thread = new thread(&sinstk::CSINSApplication::runReverseLC, this);
			}
			else
			{
				runReverseLC();
			}
		}

		// 再做一遍前向
		if ((gs_bHCDH || gs_bLJYY) && m_SINSOpt.m_FilterType == 2)
		{
			m_AppInfo[0].FilterType = 2;
			if (gs_bUsedPThread)
			{
				gs_Forward_Thread = new thread(&sinstk::CSINSApplication::runForwardLC, this);
			}
			else
			{
				runForwardLC();
			}
		}

		if (m_SINSOpt.m_bRTS && m_SINSOpt.m_FilterType == 2)
		{
			// 等待线程结束,再往下继续运行
			if (gs_bUsedPThread)
			{
				gs_Forward_Thread->join();
				gs_Reverse_Thread->join();

				delete gs_Forward_Thread; gs_Forward_Thread = NULL;
				delete gs_Reverse_Thread; gs_Reverse_Thread = NULL;
			}

			m_AppInfo[0].bFinish = false;
			m_AppInfo[0].bValid = true;
			strcpy(m_AppInfo[0].Title, "Combination: LCI");
			if (m_SINSOpt.m_bRTS)
			SINS_Combination(m_OptFile, m_bflsFile, m_brlsFile, m_dlsFile, &m_SINSOpt, &m_AppInfo[0].progress);
			SINS_Combination(m_OptFile, m_bflfFile, m_brlfFile, m_dlfFile, &m_SINSOpt, &m_AppInfo[0].progress); // 无RTS的双向组合平滑
			m_AppInfo[0].bFinish = true;
			m_AppInfo[0].bValid = false;
		}

		if (m_SINSOpt.m_ExpXMLFile != "")
		{
			char cFileTitle[1024] = "";
			getFileInfo(_C(m_IMUFile.c_str()), cFileTitle, NULL, NULL, NULL);
			string ExportFile = m_FilePath + FILEPATHSEP + cFileTitle + ".PosT";

			string SatFile = "";
			if (gs_bHCDH)
			{
				getFileInfo(_C(m_OBSFile[0].c_str()), cFileTitle, NULL, NULL, NULL);
				SatFile = m_SINSOpt.m_GNSSOpt.m_GNSS_ResultPath + "R_RTK" + FILEPATHSEP + cFileTitle + "." + gs_GNSSFileTypes[GFT_FFP] + "_sat2";
			}

			CSINSSolutionExport SINSExportTest;
			SINSExportTest.initSINSExp(m_bdlsFile, m_IMUFile, m_SINSOpt.m_MarkFile, m_SINSOpt.m_ExpXMLFile, ExportFile, m_SINSOpt.m_RotAngleIMU, SatFile);
			SINSExportTest.ExportSolution();
		}

		return true;
	}


	bool CSINSApplication::runForwardPADLC()
	{
		ZeroStruct(m_SINSSolRpt[0], SOLRPT);
		m_AppInfo[0].bFinish = false;
		m_AppInfo[0].bValid = true;
		strcpy(m_AppInfo[0].Title, "Forward KF: PAD");
		if (gs_bUsedPThread) { DUMP_OPEN_Forward(m_flfmsgFile) }
		else { DUMP_OPEN(m_flfmsgFile) }

		bool bReverse = false;
		bool bInitELC = false;
		CSINSOption SINSOpt = m_SINSOpt;
		if (SINSOpt.m_bIMUProcAll)
		{
			SINSOpt.m_IMUBeginTime = m_IMUDataContainer.m_BeginTime;
			SINSOpt.m_IMUEndTime = m_IMUDataContainer.m_EndTime;
		}

		if (m_ELCDataContainer.m_bValid)
		{
			SINSOpt.m_ELCBeginTime = m_ELCDataContainer.m_BeginTime;
			SINSOpt.m_ELCEndTime = m_ELCDataContainer.m_EndTime;
			bInitELC = true;
		}

		GPSTIME gt, IMUTime;

		// 1. 读取SINS数据
		CIMUFFStream IMUFFStream;
		IMUFFStream.m_IMUDataGapTol = SINSOpt.m_IMUDataGapTol;
		IMUFFStream.setDataContainer(&m_IMUDataContainer);
		if (IMUFFStream.init(bReverse) == false) return false;
		IMUFFStream.getOneRecordData();
		IMUTime = IMUFFStream.m_time;
		while (IMUFFStream.peakOneRecordTime() <= SINSOpt.m_IMUBeginTime) IMUFFStream.getOneRecordData();
		if (IMUFFStream.CheckDataIsEOF()) return false;


		// 2. 读取ELC数据
		CELCFFStream ELCFFStream;
		ELCFFStream.setDataContainer(&m_ELCDataContainer);
		if (ELCFFStream.init(bReverse) == false) return false;
		ELCFFStream.getOneRecordData();
		gt = ELCFFStream.m_time;
		while (ELCFFStream.peakOneRecordTime() <= SINSOpt.m_IMUBeginTime) ELCFFStream.getOneRecordData();
		if (ELCFFStream.CheckDataIsEOF()) return false;


		// 4. 初始化LC
		if (1) // 括号里的类运行完后会自动调用析构函数,关闭相应文件
		{
			CSINSSolution SINSSol;
			SINSSol.m_pSINSOpt = &SINSOpt;
			AssignSINSHead(&SINSSol.m_SINSHead);
			if (SINSSol.init(m_flfFile, true, false) == false) return false;

			CLCPADECEF* pForwardLC = new CLCPADECEF();			
			pForwardLC->m_pSINSSol = &SINSSol;
			pForwardLC->m_pIMUFFStream = &IMUFFStream;
			pForwardLC->m_pELCFFStream = &ELCFFStream;
			pForwardLC->m_pSINSOpt = &SINSOpt;
			pForwardLC->m_MFFileName = m_bflfFile;
			if (pForwardLC->initLC(bReverse) == false) return false;
			
			// 进度条
			double data_len = fabs(SINSOpt.m_IMUEndTime - SINSOpt.m_IMUBeginTime);
			double data_i = 0;
			double fProgress = 0.0;
			int iProgress = 0, iProgressPre = 0;
			bool flag_ELC = true;
			bool flag_IMU = true;
			bool bELCData = true;
			bool bIMUData = true;
			int  errorID = 0;

			while (1)
			{
				if (flag_ELC)
				{
					if (ELCFFStream.getOneRecordData() == false) break;
					gt = ELCFFStream.m_time;
				}

				if (flag_IMU)
				{
					if (IMUFFStream.getOneRecordData(true) == false) break;
					IMUTime = IMUFFStream.m_time;
				}

				if (IMUTime > SINSOpt.m_IMUEndTime) break;

				flag_ELC = flag_IMU = bELCData = bIMUData = false;

				// 1. 判断LC的操作
				if (IMUTime - IMUFFStream.m_DeltT < gt && gt <= IMUTime)
				{
					// 执行观测更新,读取两者数据
					bELCData = true;
					flag_ELC = true;
					flag_IMU = true;
				}
				else if (gt > IMUTime)
				{
					// 执行状态更新,读取IMU数据
					bIMUData = true;
					flag_IMU = true;
				}
				else
				{
					// 不进行任何更新,读取ELC数据
					flag_ELC = true;
				}

				if (bELCData)
				{
					if (gt != IMUTime) IMUFFStream.setMeasUpdate(gt); // 观测更新时设置下进行内插
					errorID = pForwardLC->runOneLC(gt, true);
				}
				else if (bIMUData)
				{
					errorID = pForwardLC->runOneLC(IMUTime, false);
				}

				// 观测更新成功,输出有用结果
				if (errorID == 0 && bELCData)
				{
					if (gs_bIPSAPP || gs_bHCDH || gs_bLJYY) // APP界面信息
					{
						data_i = fabs(SINSSol.m_time - SINSOpt.m_IMUBeginTime);
						if (data_i < data_len)
						{
							fProgress = (data_i / data_len) * 100.0; // 进度条
							iProgress = ceil(fProgress);
						}
						SINSSol.FlushAppInfo(fProgress, &m_AppInfo[0], m_SINSSolRpt);
						if (iProgress != iProgressPre)
						{
							DumpProgress(iProgress, m_AppInfo[0]);
						}
						iProgressPre = iProgress;
					}
				}
				else if (errorID == 2) break;
			}

			// 删除,防止内存泄露,不删除将报错
			delete pForwardLC;
		}

		if (gs_bUsedPThread) { DUMP_CLOSE_Forward }
		else { DUMP_CLOSE }

		if (SINSOpt.m_bRTS)
		{
		//	LC_RTS_Smoother(false, m_OptFile, m_bflfFile, m_bflsFile, m_flsFile);
		}

		m_AppInfo[0].bFinish = true;
		m_AppInfo[0].bValid = false;

		return true;
	}


	bool CSINSApplication::runReversePADLC()
	{
		ZeroStruct(m_SINSSolRpt[1], SOLRPT);
		m_AppInfo[1].bFinish = false;
		m_AppInfo[1].bValid = true;
		strcpy(m_AppInfo[1].Title, "Reverse KF: PAD");
		if (gs_bUsedPThread) { DUMP_OPEN_Reverse(m_rlfmsgFile) }
		else { DUMP_OPEN(m_rlfmsgFile) }

		bool bReverse = true;
		bool bInitELC = false;
		CSINSOption SINSOpt = m_SINSOpt;
		if (SINSOpt.m_bIMUProcAll)
		{
			SINSOpt.m_IMUBeginTime = m_IMUDataContainer.m_BeginTime;
			SINSOpt.m_IMUEndTime = m_IMUDataContainer.m_EndTime;
		}

		if (m_ELCDataContainer.m_bValid)
		{
			SINSOpt.m_ELCBeginTime = m_ELCDataContainer.m_BeginTime;
			SINSOpt.m_ELCEndTime = m_ELCDataContainer.m_EndTime;
			bInitELC = true;
		}

		GPSTIME gt, IMUTime;

		// 1. 读取SINS数据
		CIMUFFStream IMUFFStream;
		IMUFFStream.m_IMUDataGapTol = SINSOpt.m_IMUDataGapTol;
		IMUFFStream.setDataContainer(&m_IMUDataContainer);
		if (IMUFFStream.init(bReverse) == false) return false;
		IMUFFStream.getOneRecordData();
		IMUTime = IMUFFStream.m_time;
		while (IMUFFStream.peakOneRecordTime() <= SINSOpt.m_IMUBeginTime) IMUFFStream.getOneRecordData();
		if (IMUFFStream.CheckDataIsEOF()) return false;


		// 2. 读取ELC数据
		CELCFFStream ELCFFStream;
		ELCFFStream.setDataContainer(&m_ELCDataContainer);
		if (ELCFFStream.init(bReverse) == false) return false;
		ELCFFStream.getOneRecordData();
		gt = ELCFFStream.m_time;
		while (ELCFFStream.peakOneRecordTime() <= SINSOpt.m_IMUBeginTime) ELCFFStream.getOneRecordData();
		if (ELCFFStream.CheckDataIsEOF()) return false;

		// 4. 初始化LC
		if (1) // 括号里的类运行完后会自动调用析构函数,关闭相应文件
		{
			CSINSSolution SINSSol;
			SINSSol.m_pSINSOpt = &SINSOpt;
			AssignSINSHead(&SINSSol.m_SINSHead);
			if (SINSSol.init(m_rlfFile, true, false) == false) return false;

			CLCPADECEF* pReverseLC = new CLCPADECEF();
			pReverseLC->m_pSINSSol = &SINSSol;
			pReverseLC->m_pIMUFFStream = &IMUFFStream;
			pReverseLC->m_pELCFFStream = &ELCFFStream;
			pReverseLC->m_pSINSOpt = &SINSOpt;
			pReverseLC->m_MFFileName = m_brlfFile;
			if (pReverseLC->initLC(bReverse) == false) return false;

			// 进度条
			double data_len = fabs(SINSOpt.m_IMUEndTime - SINSOpt.m_IMUBeginTime);
			double data_i = data_len;
			double fProgress = 0.0;
			int iProgress = 0, iProgressPre = 0;
			bool flag_ELC = true;
			bool flag_IMU = true;
			bool bELCData = true;
			bool bIMUData = true;
			int  errorID = 0;

			while (1)
			{
				if (flag_ELC)
				{
					if (ELCFFStream.getOneRecordData() == false) break;
					gt = ELCFFStream.m_time;
				}

				if (flag_IMU)
				{
					if (IMUFFStream.getOneRecordData(true) == false) break;
					IMUTime = IMUFFStream.m_time;
				}

				if (IMUTime < SINSOpt.m_IMUBeginTime) break;

				flag_ELC = flag_IMU = bELCData = bIMUData = false;

				// 1. 判断LC的操作
				if (IMUTime + IMUFFStream.m_DeltT > gt && gt >= IMUTime)
				{
					// 执行观测更新,读取两者数据
					bELCData = true;
					flag_ELC = true;
					flag_IMU = true;
				}
				else if (gt < IMUTime)
				{
					// 执行状态更新,读取IMU数据
					bIMUData = true;
					flag_IMU = true;
				}
				else
				{
					// 不进行任何更新,读取ELC数据
					flag_ELC = true;
				}

				if (bELCData)
				{
					if (gt != IMUTime) IMUFFStream.setMeasUpdate(gt); // 观测更新时设置下进行内插
					errorID = pReverseLC->runOneLC(gt, true);
				}
				else if (bIMUData)
				{
					errorID = pReverseLC->runOneLC(IMUTime, false);
				}

				// 观测更新成功,输出有用结果
				if (errorID == 0 && bELCData)
				{
					if (gs_bIPSAPP || gs_bHCDH || gs_bLJYY) // APP界面信息
					{
						if (gs_bHCDH || gs_bLJYY) data_i = fabs(SINSSol.m_time - SINSOpt.m_IMUEndTime);
						else if (gs_bIPSAPP) data_i = fabs(SINSSol.m_time - SINSOpt.m_IMUBeginTime);
						if (data_i < data_len)
						{
							fProgress = (data_i / data_len) * 100.0; // 进度条
							iProgress = ceil(fProgress);
						}
						SINSSol.FlushAppInfo(fProgress, &m_AppInfo[1], m_SINSSolRpt + 1);
						if (iProgress != iProgressPre)
						{
							DumpProgress(iProgress, m_AppInfo[1]);
						}
						iProgressPre = iProgress;
					}
				}
				else if (errorID == 2) break;
			}

			// 删除,防止内存泄露,不删除将报错
			delete pReverseLC;
		}

		if (gs_bUsedPThread) { DUMP_CLOSE_Reverse }
		else { DUMP_CLOSE }

		ReverseFile(m_rlfFile, 2);

		if (SINSOpt.m_bRTS)
		{
		//	LC_RTS_Smoother(true, m_OptFile, m_brlfFile, m_brlsFile, m_rlsFile);
		}

		m_AppInfo[1].bFinish = true;
		m_AppInfo[1].bValid = false;

		return true;
	}


	bool CSINSApplication::runPADLC()
	{
		if (m_SINSOpt.m_FilterType == 0 || m_SINSOpt.m_FilterType == 2)
		{
			if (gs_bUsedPThread)
			{
//				int err = pthread_create(&gs_Forward_Thread, NULL, runForwardPADLC, (void*)this);
			}
			else
			{
				runForwardPADLC();
			}
		}

		if (m_SINSOpt.m_FilterType == 1 || m_SINSOpt.m_FilterType == 2)
		{
			if (gs_bUsedPThread)
			{
//				int err = pthread_create(&gs_Reverse_Thread, NULL, runReversePADLC, (void*)this);
			}
			else
			{
				runReversePADLC();
			}
		}

		return true;
	}


	bool CSINSApplication::runForwardPPPTC()
	{
		ZeroStruct(m_SINSSolRpt[0], SOLRPT);
		m_AppInfo[0].bFinish = false;
		m_AppInfo[0].bValid = true;
		if (m_pGNSSOpt->m_PMode == 2) strcpy(m_AppInfo[0].Title, "Forward KF: PPP-TCI");
		if (m_pGNSSOpt->m_PMode == 1) strcpy(m_AppInfo[0].Title, "Forward KF: SPP-TCI");

		if (gs_bUsedPThread) { DUMP_OPEN_Forward(m_ftfmsgFile) }
		else { DUMP_OPEN(m_ftfmsgFile) }

		bool bReverse = false;
		bool bInitELC = false;
		CSINSOption SINSOpt = m_SINSOpt;
		CGNSSOption* pGNSSOpt = &(SINSOpt.m_GNSSOpt);
		pGNSSOpt->m_ResultFilePath = m_ftfFile;
		if (m_pGNSSOpt->m_PMode == 1)
		{
			SINSOpt.m_bPHSUsed = false;
			pGNSSOpt->m_PPPEph = EPH_BRDC;
		}

		if (SINSOpt.m_bIMUProcAll)
		{
			SINSOpt.m_IMUBeginTime = m_IMUDataContainer.m_BeginTime;
			SINSOpt.m_IMUEndTime = m_IMUDataContainer.m_EndTime;
		}

		if (m_ELCDataContainer.m_bValid)
		{
			SINSOpt.m_ELCBeginTime = m_ELCDataContainer.m_BeginTime;
			SINSOpt.m_ELCEndTime = m_ELCDataContainer.m_EndTime;
			bInitELC = true;
		}

		// 只有ELC可用时进行SINS初始化对准,获取初始位置速度和姿态
		if (bInitELC && m_SINSInitState.init(bReverse, &SINSOpt, &m_ELCDataContainer, &m_IMUDataContainer))
		{
			if (!m_SINSInitState.getSINSInitState()) return false;
		}
		else
		{
			string msg("There is no ELC file, fail to get SINS init state!");
			DumpException(msg, LOGT_WARNING | LOGT_AFX, DUMP_LOCATION);
			return false;
		}

		GPSTIME gt, IMUTime;

		// 1. 读取GNSS数据
		COBSFFStream OBSFFStream;
		OBSFFStream.setDataContainer(&m_OBSDataContainers[0]);
		if (OBSFFStream.init(bReverse) == false) return false;
		OBSFFStream.getOneRecordData();
		gt = OBSFFStream.m_time;
		while (OBSFFStream.peakOneRecordTime() <= SINSOpt.m_IMUBeginTime) OBSFFStream.getOneRecordData();
		if (OBSFFStream.CheckDataIsEOF()) return false;


		// 2. 读取SINS数据
		CIMUFFStream IMUFFStream;
		IMUFFStream.m_IMUDataGapTol = SINSOpt.m_IMUDataGapTol;
		IMUFFStream.setDataContainer(&m_IMUDataContainer);
		if (IMUFFStream.init(bReverse) == false) return false;
		IMUFFStream.getOneRecordData();
		IMUTime = IMUFFStream.m_time;
		while (IMUFFStream.peakOneRecordTime() <= SINSOpt.m_IMUBeginTime) IMUFFStream.getOneRecordData();
		if (IMUFFStream.CheckDataIsEOF()) return false;


		// 3. 读取ELC数据
		CELCFFStream ELCFFStream;
		ELCFFStream.setDataContainer(&m_ELCDataContainer);
		if (ELCFFStream.init(bReverse) == false) return false;
		ELCFFStream.getOneRecordData();
		gt = ELCFFStream.m_time;
		while (ELCFFStream.peakOneRecordTime() <= SINSOpt.m_IMUBeginTime) ELCFFStream.getOneRecordData();
		if (ELCFFStream.CheckDataIsEOF()) return false;


		// 4. 初始化TC
		if (1) // 括号里的类运行完后会自动调用析构函数,关闭相应文件
		{
			CSINSSolution SINSSol;
			SINSSol.m_pSINSOpt = &SINSOpt;
			AssignSINSHead(&SINSSol.m_SINSHead);
			if (SINSSol.init(m_ftfFile, true) == false) return false;

			CPPPTCIFLC ForwardTC;
			ForwardTC.m_pSINSOpt = &SINSOpt;
			ForwardTC.m_pSINSSol = &SINSSol;
			ForwardTC.m_pIMUFFStream = &IMUFFStream;
			ForwardTC.m_pOBSFFStream = &OBSFFStream;
			ForwardTC.m_pEphStore = &m_EphStores;
			ForwardTC.m_pIGSProducts = &m_IGSProducts;
			if (bInitELC) ForwardTC.m_pELCFFStream = &ELCFFStream;
			else          ForwardTC.m_pELCFFStream = NULL;
			if (ForwardTC.initTC(bReverse) == false) return false;

			// 进度条
			double data_len = fabs(SINSOpt.m_IMUEndTime - SINSOpt.m_IMUBeginTime);
			double data_i = 0;
			double fProgress = 0.0;
			int iProgress = 0, iProgressPre = 0;
			bool flag_OBS = true;
			bool flag_ELC = true;
			bool flag_IMU = true;
			bool bOBSData = false;
			bool bELCData = false;
			bool bIMUData = false;
			int  errorID = 0;
			GPSTIME gt_OBS;
			GPSTIME gt_ELC;

			while (1)
			{
				if (bInitELC && flag_ELC)
				{
					if (ELCFFStream.getOneRecordData() == false) break;
					gt_ELC = ELCFFStream.m_time;
				}

				if (flag_OBS)
				{
					if (OBSFFStream.getOneRecordData() == false) break;
					gt_OBS = OBSFFStream.m_time;
				}

				if (flag_IMU)
				{
					if (IMUFFStream.getOneRecordData(true) == false) break;
					IMUTime = IMUFFStream.m_time;
				}

				if (IMUTime > SINSOpt.m_IMUEndTime) break;

				flag_OBS = flag_ELC = flag_IMU = false;
				bOBSData = bELCData = bELCData = false;

				if (IMUTime - IMUFFStream.m_DeltT < gt_OBS && gt_OBS <= IMUTime)
				{
					// 执行观测更新,读取OBS+IMU数据
					bOBSData = true;
					flag_OBS = true;
					flag_IMU = true;
				}
				else if (gt_OBS > IMUTime)
				{
					// 执行状态更新,读取IMU数据
					bIMUData = true;
					flag_IMU = true;
				}
				else
				{
					// 不进行任何更新,读取OBS数据
					flag_OBS = true;
				}

				if (bInitELC)
				{
					if (IMUTime - IMUFFStream.m_DeltT < gt_ELC && gt_ELC <= IMUTime)
					{
						// 执行观测更新,读取ELC+IMU数据
						bELCData = true;
						flag_ELC = true;
						flag_IMU = true;
					}
					else if (gt_ELC > IMUTime)
					{
						// 执行状态更新,读取IMU数据
						bIMUData = true;
						flag_IMU = true;
					}
					else
					{
						// 不进行任何更新,读取ELC数据
						flag_ELC = true;
					}
				}
				
				if (bOBSData || bELCData)
				{
					// 如果同时存在,gt取gt_OBS
					if (bELCData) gt = gt_ELC;
					if (bOBSData) gt = gt_OBS;

					if (gt != IMUTime) IMUFFStream.setMeasUpdate(gt); // 观测更新时设置下进行内插
					errorID = ForwardTC.runOneTC(gt, bOBSData, bELCData);
				}
				else if (bIMUData)
				{
					errorID = ForwardTC.runOneTC(IMUTime, false, false);
				}

				// 观测更新成功,输出有用结果
				if (errorID == 0 && (bOBSData || bELCData))
				{
					if (gs_bIPSAPP || gs_bHCDH || gs_bLJYY) // APP界面信息
					{
						data_i = fabs(SINSSol.m_time - SINSOpt.m_IMUBeginTime);
						if (data_i < data_len)
						{
							fProgress = (data_i / data_len) * 100.0; // 进度条
							iProgress = ceil(fProgress);
						}
						SINSSol.FlushAppInfo(fProgress, &m_AppInfo[0], m_SINSSolRpt);
						if (iProgress != iProgressPre)
						{
							DumpProgress(iProgress, m_AppInfo[0]);
						}
						iProgressPre = iProgress;
					}
				}
				else if (errorID == 2) break;
			}

			// 作为下次后向滤波的初始对准结果
			m_SINSInitState.getNxtSINSInitState(SINSSol.m_time, SINSSol.m_XYZPos, SINSSol.m_XYZVel, SINSSol.m_Att, SINSSol.m_GyrosBias, SINSSol.m_AccelBias);
		}
					
		if (gs_bUsedPThread) { DUMP_CLOSE_Forward }
		else { DUMP_CLOSE }
		
		if (SINSOpt.m_bRTS)
		{
			if (m_pGNSSOpt->m_PMode == 2) strcpy(m_AppInfo[0].Title, "Forward RTS: PPP-TC");
			if (m_pGNSSOpt->m_PMode == 1) strcpy(m_AppInfo[0].Title, "Forward RTS: SPP-TC");
			SINS_RTS_Smoother(false, m_OptFile, m_bftfFile, m_ftsFile, &m_SINSOpt, &m_AppInfo[0].progress);
		}

		m_AppInfo[0].bFinish = true;
		m_AppInfo[0].bValid = false;

		return true;
	}


	bool CSINSApplication::runReversePPPTC()
	{
		ZeroStruct(m_SINSSolRpt[1], SOLRPT);
		m_AppInfo[1].bFinish = false;
		m_AppInfo[1].bValid = true;
		if (m_pGNSSOpt->m_PMode == 2) strcpy(m_AppInfo[1].Title, "Reverse KF: PPP-TC");
		if (m_pGNSSOpt->m_PMode == 1) strcpy(m_AppInfo[1].Title, "Reverse KF: SPP-TC");

		if (gs_bUsedPThread) { DUMP_OPEN_Reverse(m_rtfmsgFile) }
		else { DUMP_OPEN(m_rtfmsgFile) }

		bool bReverse = true;
		bool bInitELC = false;
		CSINSOption SINSOpt = m_SINSOpt;
		CGNSSOption* pGNSSOpt = &(SINSOpt.m_GNSSOpt);
		pGNSSOpt->m_ResultFilePath = m_rtfFile;
		pGNSSOpt->m_BaseSiteID = 1; // 默认使用第一个基站,未来多基站处理需要改
		if (m_pGNSSOpt->m_PMode == 1)
		{
			SINSOpt.m_bPHSUsed = false;
			pGNSSOpt->m_PPPEph = EPH_BRDC;
		}

		if (SINSOpt.m_bIMUProcAll)
		{
			SINSOpt.m_IMUBeginTime = m_IMUDataContainer.m_BeginTime;
			SINSOpt.m_IMUEndTime = m_IMUDataContainer.m_EndTime;
		}

		if (m_ELCDataContainer.m_bValid)
		{
			SINSOpt.m_ELCBeginTime = m_ELCDataContainer.m_BeginTime;
			SINSOpt.m_ELCEndTime = m_ELCDataContainer.m_EndTime;
			bInitELC = true;
		}

		// 只有ELC可用时进行SINS初始化对准,获取初始位置速度和姿态
		if (bInitELC && m_SINSInitState.init(bReverse, &SINSOpt, &m_ELCDataContainer, &m_IMUDataContainer))
		{
			if (!m_SINSInitState.getSINSInitState()) return false;
		}
		else
		{
			string msg("There is no ELC file, fail to get SINS init state!");
			DumpException(msg, LOGT_WARNING | LOGT_AFX, DUMP_LOCATION);
			return false;
		}

		GPSTIME gt, IMUTime;

		// 1. 读取GNSS数据
		COBSFFStream OBSFFStream;
		OBSFFStream.setDataContainer(&m_OBSDataContainers[0]);
		if (OBSFFStream.init(bReverse) == false) return false;
		OBSFFStream.getOneRecordData();
		gt = OBSFFStream.m_time;
		while (OBSFFStream.peakOneRecordTime() >= SINSOpt.m_IMUEndTime) OBSFFStream.getOneRecordData();
		if (OBSFFStream.CheckDataIsEOF()) return false;


		// 2. 读取SINS数据
		CIMUFFStream IMUFFStream;
		IMUFFStream.m_IMUDataGapTol = SINSOpt.m_IMUDataGapTol;
		IMUFFStream.setDataContainer(&m_IMUDataContainer);
		if (IMUFFStream.init(bReverse) == false) return false;
		IMUFFStream.getOneRecordData();
		IMUTime = IMUFFStream.m_time;
		while (IMUFFStream.peakOneRecordTime() >= SINSOpt.m_IMUEndTime) IMUFFStream.getOneRecordData();
		if (IMUFFStream.CheckDataIsEOF()) return false;

		// 只是为了与原来的对齐,可以去掉???
		// IMUFFStream.getOneRecordData();

		// 3. 读取ELC数据
		CELCFFStream ELCFFStream;
		ELCFFStream.setDataContainer(&m_ELCDataContainer);
		if (ELCFFStream.init(bReverse) == false) return false;
		ELCFFStream.getOneRecordData();
		gt = ELCFFStream.m_time;
		while (ELCFFStream.peakOneRecordTime() >= SINSOpt.m_IMUEndTime) ELCFFStream.getOneRecordData();
		if (ELCFFStream.CheckDataIsEOF()) return false;


		// 4. 初始化TC
		if (1) // 括号里的类运行完后会自动调用析构函数,关闭相应文件
		{
			CSINSSolution SINSSol;
			SINSSol.m_pSINSOpt = &SINSOpt;
			AssignSINSHead(&SINSSol.m_SINSHead);
			if (SINSSol.init(m_rtfFile, true) == false) return false;

			CPPPTCIFLC ReverseTC;
			ReverseTC.m_pSINSOpt = &SINSOpt;
			ReverseTC.m_pSINSSol = &SINSSol;
			ReverseTC.m_pIMUFFStream = &IMUFFStream;
			ReverseTC.m_pOBSFFStream = &OBSFFStream;
			ReverseTC.m_pEphStore = &m_EphStores;
			ReverseTC.m_pIGSProducts = &m_IGSProducts;
			if (bInitELC) ReverseTC.m_pELCFFStream = &ELCFFStream;
			else          ReverseTC.m_pELCFFStream = NULL;
			if (ReverseTC.initTC(bReverse) == false) return false;

			// 进度条
			double data_len = fabs(SINSOpt.m_IMUEndTime - SINSOpt.m_IMUBeginTime);
			double data_i = data_len;
			double fProgress = 0.0;
			int iProgress = 0, iProgressPre = 0;
			bool flag_OBS = true;
			bool flag_ELC = true;
			bool flag_IMU = true;
			bool bOBSData = false;
			bool bELCData = false;
			bool bIMUData = false;
			int  errorID = 0;
			GPSTIME gt_OBS;
			GPSTIME gt_ELC;

			while (1)
			{
				if (bInitELC && flag_ELC)
				{
					if (ELCFFStream.getOneRecordData() == false) break;
					gt_ELC = ELCFFStream.m_time;
				}

				if (flag_OBS)
				{
					if (OBSFFStream.getOneRecordData() == false) break;
					gt_OBS = OBSFFStream.m_time;
				}

				if (flag_IMU)
				{
					if (IMUFFStream.getOneRecordData(true) == false) break;
					IMUTime = IMUFFStream.m_time;
				}

				if (IMUTime < SINSOpt.m_IMUBeginTime) break;

				flag_OBS = flag_ELC = flag_IMU = false;
				bOBSData = bELCData = bELCData = false;

				if (IMUTime + IMUFFStream.m_DeltT > gt_OBS && gt_OBS >= IMUTime)
				{
					// 执行观测更新,读取OBS+IMU数据
					bOBSData = true;
					flag_OBS = true;
					flag_IMU = true;
				}
				else if (gt_OBS < IMUTime)
				{
					// 执行状态更新,读取IMU数据
					bIMUData = true;
					flag_IMU = true;
				}
				else
				{
					// 不进行任何更新,读取OBS数据
					flag_OBS = true;
				}

				if (bInitELC)
				{
					if (IMUTime + IMUFFStream.m_DeltT > gt_ELC && gt_ELC >= IMUTime)
					{
						// 执行观测更新,读取ELC+IMU数据
						bELCData = true;
						flag_ELC = true;
						flag_IMU = true;
					}
					else if (gt_ELC < IMUTime)
					{
						// 执行状态更新,读取IMU数据
						bIMUData = true;
						flag_IMU = true;
					}
					else
					{
						// 不进行任何更新,读取ELC数据
						flag_ELC = true;
					}
				}

				if (bOBSData || bELCData)
				{
					// 如果同时存在,gt取gt_OBS
					if (bELCData) gt = gt_ELC;
					if (bOBSData) gt = gt_OBS;

					if (gt != IMUTime) IMUFFStream.setMeasUpdate(gt); // 观测更新时设置下进行内插
					errorID = ReverseTC.runOneTC(gt, bOBSData, bELCData);
				}
				else if (bIMUData)
				{
					errorID = ReverseTC.runOneTC(IMUTime, false, false);
				}

				// 观测更新成功,输出有用结果
				if (errorID == 0 && (bOBSData || bELCData))
				{
					if (gs_bIPSAPP || gs_bHCDH || gs_bLJYY) // APP界面信息
					{
						if (gs_bHCDH || gs_bLJYY) data_i = fabs(SINSSol.m_time - SINSOpt.m_IMUEndTime);
						else if (gs_bIPSAPP) data_i = fabs(SINSSol.m_time - SINSOpt.m_IMUBeginTime);
						if (data_i < data_len)
						{
							fProgress = (data_i / data_len) * 100.0; // 进度条
							iProgress = ceil(fProgress);
						}
						SINSSol.FlushAppInfo(fProgress, &m_AppInfo[1], m_SINSSolRpt + 1);
						if (iProgress != iProgressPre)
						{
							DumpProgress(iProgress, m_AppInfo[1]);
						}
						iProgressPre = iProgress;
					}
				}
				else if (errorID == 2) break;

				// 作为下次前向滤波的初始对准结果
				m_SINSInitState.getNxtSINSInitState(SINSSol.m_time, SINSSol.m_XYZPos, SINSSol.m_XYZVel, SINSSol.m_Att, SINSSol.m_GyrosBias, SINSSol.m_AccelBias);
			}
		}
		
		if (gs_bUsedPThread) { DUMP_CLOSE_Reverse }
		else { DUMP_CLOSE }

		ReverseFile(m_rtfFile, 2);
		
		if (SINSOpt.m_bRTS)
		{
			if (m_pGNSSOpt->m_PMode == 2) strcpy(m_AppInfo[1].Title, "Reverse RTS: PPP-TC");
			if (m_pGNSSOpt->m_PMode == 1) strcpy(m_AppInfo[1].Title, "Reverse RTS: SPP-TC");
			SINS_RTS_Smoother(true, m_OptFile, m_brtfFile, m_rtsFile, &m_SINSOpt, &m_AppInfo[1].progress);
		}

		m_AppInfo[1].bFinish = true;
		m_AppInfo[1].bValid = false;

		return true;
	}


	void CSINSApplication::runPPPTCPostIAR()
	{
		bool bRTS = m_SINSOpt.m_bRTS;
		m_SINSOpt.m_bRTS = false;
		m_AmbIntegration.m_pGNSSOpt = m_pGNSSOpt;

		m_bAmbIntegration = false;
		runForwardPPPTC();
		runReversePPPTC();

		string ambfile;
		ambfile = m_ftfFile + "_amb";
		m_AmbIntegration.readIARMsgFile(false, ambfile);
		ambfile = m_rtfFile + "_amb";
		m_AmbIntegration.readIARMsgFile(true, ambfile);
		m_AmbIntegration.runAmbAnalyst();

		m_bAmbIntegration = true;
		m_SINSOpt.m_bRTS = bRTS;

		m_AmbIntegration.m_bReverse = false;
		runForwardPPPTC();
		m_AmbIntegration.m_bReverse = true;
		runReversePPPTC();

		m_AppInfo[0].bFinish = false;
		m_AppInfo[0].bValid = true;
		strcpy(m_AppInfo[0].Title, "Combination: TCI");
		if (m_SINSOpt.m_bRTS)
		SINS_Combination(m_OptFile, m_bftsFile, m_brtsFile, m_dtsFile, &m_SINSOpt, &m_AppInfo[0].progress);
		SINS_Combination(m_OptFile, m_bftfFile, m_brtfFile, m_dtfFile, &m_SINSOpt, &m_AppInfo[0].progress); // 无RTS的双向组合平滑
		m_AppInfo[0].bFinish = true;
		m_AppInfo[0].bValid = false;
	}


	void CSINSApplication::runRTKTCPostIAR()
	{
		bool bRTS = m_SINSOpt.m_bRTS;
		m_SINSOpt.m_bRTS = false;
		m_AmbIntegration.m_pGNSSOpt = m_pGNSSOpt;

		m_bAmbIntegration = false;
		runForwardRTKTC();
		runReverseRTKTC();

		string ambfile;
		ambfile = m_ftfFile + "_amb";
		m_AmbIntegration.readIARMsgFile(false, ambfile);
		ambfile = m_rtfFile + "_amb";
		m_AmbIntegration.readIARMsgFile(true, ambfile);
		m_AmbIntegration.runAmbAnalyst();

		m_bAmbIntegration = true;
		m_SINSOpt.m_bRTS = bRTS;

		m_AmbIntegration.m_bReverse = false;
		runForwardRTKTC();
		m_AmbIntegration.m_bReverse = true;
		runReverseRTKTC();

		m_AppInfo[0].bFinish = false;
		m_AppInfo[0].bValid = true;
		strcpy(m_AppInfo[0].Title, "Combination: TCI");
		if (m_SINSOpt.m_bRTS)
			SINS_Combination(m_OptFile, m_bftsFile, m_brtsFile, m_dtsFile, &m_SINSOpt, &m_AppInfo[0].progress);
		SINS_Combination(m_OptFile, m_bftfFile, m_brtfFile, m_dtfFile, &m_SINSOpt, &m_AppInfo[0].progress); // 无RTS的双向组合平滑
		m_AppInfo[0].bFinish = true;
		m_AppInfo[0].bValid = false;
	}


	bool CSINSApplication::MatchRTKData(COBSFFStream& BaseOBSFFStream, COBSFFStream& RoveOBSFFStream, bool bReverse)
	{
		// m_SampleTime >= 0 时,按照流动站采样率处理
		if (m_pGNSSOpt->m_ProcRate >= 0)
		{
			if (RoveOBSFFStream.getOneRecordData() == false) return false;
			if (BaseOBSFFStream.m_time.GPSWeek < 10 && BaseOBSFFStream.getOneRecordData() == false) return false;

			GPSTIME gtNxt, gtCur;
			double dt0, dt1;

			while (1)
			{
				if (RoveOBSFFStream.CheckDataIsEOF() || BaseOBSFFStream.CheckDataIsEOF()) return false;

				gtNxt = BaseOBSFFStream.peakOneRecordTime();
				if (gtNxt.GPSWeek < 1) break; // 当时间没有时结束

				gtCur = BaseOBSFFStream.m_time;

				if (bReverse)
				{
					dt0 = gtCur - RoveOBSFFStream.m_time;
					dt1 = RoveOBSFFStream.m_time - gtNxt;
				}
				else
				{
					dt0 = RoveOBSFFStream.m_time - gtCur;
					dt1 = gtNxt - RoveOBSFFStream.m_time;
				}

				// 以下两种方式都可以
				// 第一种起始时间是和基站对齐的, 第二种只要流动站与基站相隔DTTOL就开始解算了
				// 第一种和事后模拟实时是对应的,如果要对比事后和实时结果,用第一种
				// 但是第一种注释的地方有问题,比如0.6s时,基站会读取下,这样0.8s的数据就不会处理了
				if (false)
				{
					if (dt0 >= 0.0 && dt1 >= 0)
					{
						if (fabs(dt0) <= fabs(dt1) && dt0 <= DTTOL_M)
						{
							return true;
						}
						else if (fabs(dt0) >= fabs(dt1) && dt1 <= DTTOL_M)
						{
						//	BaseOBSFFStream.getOneRecordData(); // 有问题
							return true;
						}
						else
						{
							RoveOBSFFStream.getOneRecordData();
						}
					}
					else if (dt0 < 0)
					{
						RoveOBSFFStream.getOneRecordData();
					}
					else if (dt1 < 0)
					{
						BaseOBSFFStream.getOneRecordData();
					}
				}
				else
				{
					if (dt0 < -DTTOL_M)
					{
						RoveOBSFFStream.getOneRecordData();
					}
					else if (dt0 <= 0)
					{
						return true;
					}
					else if (dt1 < 0)
					{
						BaseOBSFFStream.getOneRecordData();
					}
					else if (fabs(dt0) <= fabs(dt1) && dt0 <= DTTOL_M)
					{
						return true;
					}
					else if (fabs(dt0) >= fabs(dt1) && dt1 <= DTTOL_M)
					{
						BaseOBSFFStream.getOneRecordData();
						return true;
					}
					else
					{
						RoveOBSFFStream.getOneRecordData();
					}
				}
			}

			return false;
		}
		else // m_SampleTime < 0 时,基站和流动站同个时间点处理
		{
			if (RoveOBSFFStream.getOneRecordData() == false || BaseOBSFFStream.getOneRecordData() == false) return false;

			double dt = RoveOBSFFStream.m_OBSData.gt - BaseOBSFFStream.m_OBSData.gt;
			if (fabs(dt) < DTTOL) return true; // 时间容忍度需要注意???

			while (1)
			{
				if (RoveOBSFFStream.CheckDataIsEOF() || BaseOBSFFStream.CheckDataIsEOF()) return false;

				dt = RoveOBSFFStream.m_OBSData.gt - BaseOBSFFStream.m_OBSData.gt;

				if (dt >= DTTOL)
				{
					if (bReverse) RoveOBSFFStream.getOneRecordData();
					else          BaseOBSFFStream.getOneRecordData();
				}
				else if (dt <= -DTTOL)
				{
					if (bReverse) BaseOBSFFStream.getOneRecordData();
					else          RoveOBSFFStream.getOneRecordData();
				}
				else
				{
					return true;
				}
			}

			return false;
		}
	}


	bool CSINSApplication::runForwardRTKTC()
	{
		ZeroStruct(m_SINSSolRpt[0], SOLRPT);
		m_AppInfo[0].bFinish = false;
		m_AppInfo[0].bValid = true;
		strcpy(m_AppInfo[0].Title, "Forward KF: RTK-TCI");

		if (gs_bUsedPThread) { DUMP_OPEN_Forward(m_ftfmsgFile) }
		else { DUMP_OPEN(m_ftfmsgFile) }

		bool bReverse = false;
		bool bInitELC = false;
		CSINSOption SINSOpt = m_SINSOpt;
		CGNSSOption* pGNSSOpt = &(SINSOpt.m_GNSSOpt);
		pGNSSOpt->m_ResultFilePath = m_ftfFile;
		pGNSSOpt->m_BaseSiteID = 1; // 默认使用第一个基站,未来多基站处理需要改
		if (SINSOpt.m_bIMUProcAll)
		{
			SINSOpt.m_IMUBeginTime = m_IMUDataContainer.m_BeginTime;
			SINSOpt.m_IMUEndTime = m_IMUDataContainer.m_EndTime;
		}

		if (m_ELCDataContainer.m_bValid)
		{
			SINSOpt.m_ELCBeginTime = m_ELCDataContainer.m_BeginTime;
			SINSOpt.m_ELCEndTime = m_ELCDataContainer.m_EndTime;
			bInitELC = true;
		}

		// 只有ELC可用时进行SINS初始化对准,获取初始位置速度和姿态
		if (bInitELC && m_SINSInitState.init(bReverse, &SINSOpt, &m_ELCDataContainer, &m_IMUDataContainer))
		{
			if (!m_SINSInitState.getSINSInitState()) return false;
		}
		else
		{
			string msg("There is no ELC file, fail to get SINS init state!");
			DumpException(msg, LOGT_WARNING | LOGT_AFX, DUMP_LOCATION);
			return false;
		}

		GPSTIME gt, IMUTime;

		// 1. 读取GNSS数据
		COBSFFStream RoveOBSFFStream;
		RoveOBSFFStream.setDataContainer(&m_OBSDataContainers[0]);
		if (RoveOBSFFStream.init(bReverse) == false) return false;
		RoveOBSFFStream.getOneRecordData();
		gt = RoveOBSFFStream.m_time;
		while (RoveOBSFFStream.peakOneRecordTime() <= SINSOpt.m_IMUBeginTime) RoveOBSFFStream.getOneRecordData();
		if (RoveOBSFFStream.CheckDataIsEOF()) return false;

		COBSFFStream BaseOBSFFStream;
		BaseOBSFFStream.setDataContainer(&m_OBSDataContainers[pGNSSOpt->m_BaseSiteID]);
		if (BaseOBSFFStream.init(bReverse) == false) return false;
		BaseOBSFFStream.getOneRecordData();
		gt = BaseOBSFFStream.m_time;
		while (BaseOBSFFStream.peakOneRecordTime() <= SINSOpt.m_IMUBeginTime) BaseOBSFFStream.getOneRecordData();
		if (BaseOBSFFStream.CheckDataIsEOF()) return false;


		// 2. 读取SINS数据
		CIMUFFStream IMUFFStream;
		IMUFFStream.m_IMUDataGapTol = SINSOpt.m_IMUDataGapTol;
		IMUFFStream.setDataContainer(&m_IMUDataContainer);
		if (IMUFFStream.init(bReverse) == false) return false;
		IMUFFStream.getOneRecordData();
		IMUTime = IMUFFStream.m_time;
		while (IMUFFStream.peakOneRecordTime() <= SINSOpt.m_IMUBeginTime) IMUFFStream.getOneRecordData();
		if (IMUFFStream.CheckDataIsEOF()) return false;


		// 3. 读取ELC数据
		CELCFFStream ELCFFStream;
		ELCFFStream.setDataContainer(&m_ELCDataContainer);
		if (ELCFFStream.init(bReverse) == false) return false;
		ELCFFStream.getOneRecordData();
		gt = ELCFFStream.m_time;
		while (ELCFFStream.peakOneRecordTime() <= SINSOpt.m_IMUBeginTime) ELCFFStream.getOneRecordData();
		if (ELCFFStream.CheckDataIsEOF()) return false;

		
		// 4. 初始化TC
		if (1) // 括号里的类运行完后会自动调用析构函数,关闭相应文件
		{
			CSINSSolution SINSSol;
			SINSSol.m_pSINSOpt = &SINSOpt;
			AssignSINSHead(&SINSSol.m_SINSHead);
			if (SINSSol.init(m_ftfFile, true) == false) return false;
			
			///////////////////////////////////////////////////////////////////////////////////////////

			CSwitchVRS SwitchVRS;
			SwitchVRS.init(pGNSSOpt, &m_IGSProducts, false);

			// step 2: 根据基线类型确定RTK模型
			CRTKTCBase* pRTKTC = NULL;
			switch (pGNSSOpt->m_RTKType)
			{
			case 0: pRTKTC = new CRTKTCSBL(); break;
			case 1: pRTKTC = new CRTKTCLBL(); break;
			case 2: pRTKTC = new CRTKTCIBL(); break;
			default: return false;
			}

			// step 3: 准备RTK的数据
			pRTKTC->m_pOBSFFStream[0] = &RoveOBSFFStream;
			pRTKTC->m_pOBSFFStream[1] = &BaseOBSFFStream;
			pRTKTC->m_pEphStore = &m_EphStores;
			pRTKTC->m_pSINSSol = &SINSSol;
			pRTKTC->m_pSINSOpt = &SINSOpt;
			pRTKTC->m_pIGSProducts = &m_IGSProducts;
			pRTKTC->m_pIMUFFStream = &IMUFFStream;
			if (bInitELC) pRTKTC->m_pELCFFStream = &ELCFFStream;
			else          pRTKTC->m_pELCFFStream = NULL;
			if (m_bMultiBaseSite) pGNSSOpt->m_BaseSiteID = BaseOBSFFStream.m_OBSData.BaseSiteID;
			if (pRTKTC->initTC(bReverse) == false) return false;

#ifdef _TwiceProc_LC_ON_

			CIMUFFStream IMUFFStreamTmp = IMUFFStream;
			CSINSSolution SINSSolTmp;
			CLooseCouplingBase* pForwardLCTmp = NULL;
			GPSTIME IMUTimeTmp = IMUFFStreamTmp.m_time;

			if (ls_bCheckScene)
			{
				if (SINSOpt.m_NaviSys == NAVI_E) pForwardLCTmp = new CLooseCouplingECEF();
				else                             pForwardLCTmp = new CLooseCouplingLocal();
				pForwardLCTmp->m_pSINSSol = &SINSSolTmp;
				pForwardLCTmp->m_pIMUFFStream = &IMUFFStreamTmp;
				pForwardLCTmp->m_pELCFFStream = &ELCFFStream;
				pForwardLCTmp->m_pSINSOpt = &SINSOpt;
				pForwardLCTmp->m_pSINSOpt->m_PMode = 2;
				if (pForwardLCTmp->initLC(bReverse) == false) return false;
				pForwardLCTmp->m_pSINSOpt->m_PMode = 3;
			}

			bool flag_IMUTmp = true;

#endif // _TwiceProc_LC_ON_

			// 进度条
			double data_len = fabs(SINSOpt.m_IMUEndTime - SINSOpt.m_IMUBeginTime);
			double data_i = 0;
			double fProgress = 0.0;
			int iProgress = 0, iProgressPre = 0;
			bool flag_OBS = true;
			bool flag_ELC = true;
			bool flag_IMU = true;
			bool bOBSData = false;
			bool bELCData = false;
			bool bIMUData = false;
			int  errorID = 0;
			GPSTIME gt_OBS;
			GPSTIME gt_ELC;

			while (1)
			{
				if (bInitELC && flag_ELC)
				{
					if (ELCFFStream.getOneRecordData() == false) break;
					gt_ELC = ELCFFStream.m_time;
				}

				if (flag_OBS)
				{
					if (MatchRTKData(BaseOBSFFStream, RoveOBSFFStream, bReverse) == false) break;
					gt_OBS = RoveOBSFFStream.m_time;
				}

				if (flag_IMU)
				{
					if (IMUFFStream.getOneRecordData(true) == false) break;
					IMUTime = IMUFFStream.m_time;
				}

#ifdef _TwiceProc_LC_ON_
				if (ls_bCheckScene)
				{
					if (flag_IMUTmp)
					{
						if (IMUFFStreamTmp.getOneRecordData(true) == false) break;
						IMUTimeTmp = IMUFFStreamTmp.m_time;
					}
				}

				flag_IMUTmp = false;

				if (gt_OBS > IMUTimeTmp)
				{
					bIMUData = true;
					flag_IMUTmp = true;
				}
#endif					

				if (IMUTime > SINSOpt.m_IMUEndTime) break;

				flag_OBS = flag_ELC = flag_IMU = false;
				bOBSData = bELCData = bIMUData = false;

				if (IMUTime - IMUFFStream.m_DeltT < gt_OBS && gt_OBS <= IMUTime)
				{
					// 执行观测更新,读取OBS+IMU数据
					bOBSData = true;
					flag_OBS = true;
					flag_IMU = true;
				}
				else if (gt_OBS > IMUTime)
				{
					// 执行状态更新,读取IMU数据
					bIMUData = true;
					flag_IMU = true;
				}
				else
				{
					// 不进行任何更新,读取OBS数据
					flag_OBS = true;
				}

				if (bInitELC)
				{
					if (IMUTime - IMUFFStream.m_DeltT < gt_ELC && gt_ELC <= IMUTime)
					{
						// 执行观测更新,读取ELC+IMU数据
						bELCData = true;
						flag_ELC = true;
						flag_IMU = true;
					}
					else if (gt_ELC > IMUTime)
					{
						// 执行状态更新,读取IMU数据
						bIMUData = true;
						flag_IMU = true;
					}
					else
					{
						// 不进行任何更新,读取ELC数据
						flag_ELC = true;
					}
				}

				if (bOBSData || bELCData)
				{
					// 如果同时存在,gt取gt_OBS
					if (bELCData) gt = gt_ELC;
					if (bOBSData) gt = gt_OBS;

					if (gt != IMUTime) IMUFFStream.setMeasUpdate(gt); // 观测更新时设置下进行内插

					// 给VRS赋值并运行
					if (bOBSData && m_bMultiBaseSite)
					{
						pGNSSOpt->m_BaseSiteID = BaseOBSFFStream.m_OBSData.BaseSiteID;
						M31EQU(pGNSSOpt->m_BaseXYZ[pGNSSOpt->m_BaseSiteID], SwitchVRS.m_VRSXYZ);
						SwitchVRS.m_VRSOBSData = BaseOBSFFStream.m_OBSData;
						SwitchVRS.m_EphStoresPostProc = &m_EphStores;
						SwitchVRS.runOneSwitchVRS();
						pRTKTC->m_bSwitchVRS = SwitchVRS.m_bSwitchVRS;
						for (int f = 0; f < NFREQ; f++) pRTKTC->m_VRSAmb[f] = SwitchVRS.m_VRSAmb[f];
					}

#ifdef _TwiceProc_LC_ON_
					if (ls_bCheckScene)
					{
						if (gt != IMUTime) IMUFFStreamTmp.setMeasUpdate(gt); // 观测更新时设置下进行内插

						int Quality = ELCFFStream.m_ELCData.GNSS_DATA.Quality; // 记录原本的Q

						pForwardLCTmp->runOneLC(gt, true);
						if ((fabs(SINSSolTmp.m_MisclosurePos[0]) > 0.2 || fabs(SINSSolTmp.m_MisclosurePos[1]) > 0.2 || fabs(SINSSolTmp.m_MisclosurePos[2]) > 0.2) && (SINSSolTmp.m_IMUFlag&ELCTYPE_POS))
						{
							Quality = 7;
							ls_nCheckPass = 0;
						}
						else if (SINSSolTmp.m_IMUFlag&ELCTYPE_POS)
						{
							ls_nCheckPass++;
						}
						else
						{
							Quality = 7;
						}

						// 检测没问题5个历元内仍不予通过
						if (SINSOpt.m_bTwiceProcLC_NumCheck && ls_nCheckPass < 5) Quality = 7;

						// 满足下列条件认为GNSS没问题，予以通过
						if ((ELCFFStream.m_ELCData.GNSS_DATA.ARRatio > 6 && ELCFFStream.m_ELCData.GNSS_DATA.nSat_ARHold > 6 && ELCFFStream.m_ELCData.GNSS_DATA.AmbState >= ARSTATE_HOLD && MatrixNorm2(3, 1, ELCFFStream.m_ELCData.GNSS_DATA.SepENU) < 0.5) ||
							(ELCFFStream.m_ELCData.GNSS_DATA.ARRatio > 10 && ELCFFStream.m_ELCData.GNSS_DATA.PDOP < 2.0 && MatrixNorm2(3, 1, ELCFFStream.m_ELCData.GNSS_DATA.SepENU) < 0.5))
							Quality = ELCFFStream.m_ELCData.GNSS_DATA.Quality;

						ELCFFStream.m_ELCData.GNSS_DATA.Quality = Quality;
					}
#endif
					errorID = pRTKTC->runOneTC(gt, bOBSData, bELCData);
				}
				else if (bIMUData)
				{
#ifdef _TwiceProc_LC_ON_
					if (ls_bCheckScene) pForwardLCTmp->runOneLC(IMUTimeTmp, false);
#endif
					errorID = pRTKTC->runOneTC(IMUTime, false, false);
				}

				// 观测更新成功,输出有用结果
				if (errorID == 0 && (bOBSData || bELCData))
				{
					if (gs_bIPSAPP || gs_bHCDH || gs_bLJYY) // APP界面信息
					{
						data_i = fabs(SINSSol.m_time - SINSOpt.m_IMUBeginTime);
						if (data_i < data_len)
						{
							fProgress = (data_i / data_len) * 100.0; // 进度条
							iProgress = ceil(fProgress);
						}
						SINSSol.FlushAppInfo(fProgress, &m_AppInfo[0], m_SINSSolRpt);
						if (iProgress != iProgressPre)
						{
							DumpProgress(iProgress, m_AppInfo[0]);
						}
						iProgressPre = iProgress;
					}
				}
				else if (errorID == 2) break;
			}

#ifdef _TwiceProc_LC_ON_
			if (pForwardLCTmp) delete pForwardLCTmp;
#endif
			delete pRTKTC;

			// 作为下次后向滤波的初始对准结果
			m_SINSInitState.getNxtSINSInitState(SINSSol.m_time, SINSSol.m_XYZPos, SINSSol.m_XYZVel, SINSSol.m_Att, SINSSol.m_GyrosBias, SINSSol.m_AccelBias);
		}

		if (gs_bUsedPThread) { DUMP_CLOSE_Forward }
		else { DUMP_CLOSE }

		if (SINSOpt.m_bRTS)
		{
			strcpy(m_AppInfo[0].Title, "Forward RTS: RTK-TCI");
			SINS_RTS_Smoother(false, m_OptFile, m_bftfFile, m_ftsFile, &m_SINSOpt, &m_AppInfo[0].progress);
		}

		m_AppInfo[0].bFinish = true;
		m_AppInfo[0].bValid = false;

		return true;
	}


	bool CSINSApplication::runReverseRTKTC()
	{
		ZeroStruct(m_SINSSolRpt[1], SOLRPT);
		m_AppInfo[1].bFinish = false;
		m_AppInfo[1].bValid = true;
		strcpy(m_AppInfo[1].Title, "Reverse KF: RTK-TCI");

		if (gs_bUsedPThread) { DUMP_OPEN_Reverse(m_rtfmsgFile) }
		else { DUMP_OPEN(m_rtfmsgFile) }

		bool bReverse = true;
		bool bInitELC = false;
		CSINSOption SINSOpt = m_SINSOpt;
		CGNSSOption* pGNSSOpt = &(SINSOpt.m_GNSSOpt);
		pGNSSOpt->m_ResultFilePath = m_rtfFile;
		pGNSSOpt->m_BaseSiteID = 1; // 默认使用第一个基站,未来多基站处理需要改
		if (SINSOpt.m_bIMUProcAll)
		{
			SINSOpt.m_IMUBeginTime = m_IMUDataContainer.m_BeginTime;
			SINSOpt.m_IMUEndTime = m_IMUDataContainer.m_EndTime;
		}

		if (m_ELCDataContainer.m_bValid)
		{
			SINSOpt.m_ELCBeginTime = m_ELCDataContainer.m_BeginTime;
			SINSOpt.m_ELCEndTime = m_ELCDataContainer.m_EndTime;
			bInitELC = true;
		}

		// 只有ELC可用时进行SINS初始化对准,获取初始位置速度和姿态
		if (bInitELC && m_SINSInitState.init(bReverse, &SINSOpt, &m_ELCDataContainer, &m_IMUDataContainer))
		{
			if (!m_SINSInitState.getSINSInitState()) return false;
		}
		else
		{
			string msg("There is no ELC file, fail to get SINS init state!");
			DumpException(msg, LOGT_WARNING | LOGT_AFX, DUMP_LOCATION);
			return false;
		}

		GPSTIME gt, IMUTime;

		// 1. 读取GNSS数据
		COBSFFStream RoveOBSFFStream;
		RoveOBSFFStream.setDataContainer(&m_OBSDataContainers[0]);
		if (RoveOBSFFStream.init(bReverse) == false) return false;
		RoveOBSFFStream.getOneRecordData();
		gt = RoveOBSFFStream.m_time;
		while (RoveOBSFFStream.peakOneRecordTime() >= SINSOpt.m_IMUEndTime) RoveOBSFFStream.getOneRecordData();
		if (RoveOBSFFStream.CheckDataIsEOF()) return false;

		COBSFFStream BaseOBSFFStream;
		BaseOBSFFStream.setDataContainer(&m_OBSDataContainers[pGNSSOpt->m_BaseSiteID]);
		if (BaseOBSFFStream.init(bReverse) == false) return false;
		BaseOBSFFStream.getOneRecordData();
		gt = BaseOBSFFStream.m_time;
		while (BaseOBSFFStream.peakOneRecordTime() >= SINSOpt.m_IMUEndTime) BaseOBSFFStream.getOneRecordData();
		if (BaseOBSFFStream.CheckDataIsEOF()) return false;


		// 2. 读取SINS数据
		CIMUFFStream IMUFFStream;
		IMUFFStream.m_IMUDataGapTol = SINSOpt.m_IMUDataGapTol;
		IMUFFStream.setDataContainer(&m_IMUDataContainer);
		if (IMUFFStream.init(bReverse) == false) return false;
		IMUFFStream.getOneRecordData();
		IMUTime = IMUFFStream.m_time;
		while (IMUFFStream.peakOneRecordTime() >= SINSOpt.m_IMUEndTime) IMUFFStream.getOneRecordData();
		if (IMUFFStream.CheckDataIsEOF()) return false;


		// 3. 读取ELC数据
		CELCFFStream ELCFFStream;
		ELCFFStream.setDataContainer(&m_ELCDataContainer);
		if (ELCFFStream.init(bReverse) == false) return false;
		ELCFFStream.getOneRecordData();
		gt = ELCFFStream.m_time;
		while (ELCFFStream.peakOneRecordTime() >= SINSOpt.m_IMUEndTime) ELCFFStream.getOneRecordData();
		if (ELCFFStream.CheckDataIsEOF()) return false;


		// 4. 初始化TC
		if (1) // 括号里的类运行完后会自动调用析构函数,关闭相应文件
		{
			CSINSSolution SINSSol;
			SINSSol.m_pSINSOpt = &SINSOpt;
			AssignSINSHead(&SINSSol.m_SINSHead);
			if (SINSSol.init(m_rtfFile, true) == false) return false;

			///////////////////////////////////////////////////////////////////////////////////////////

			CSwitchVRS SwitchVRS;
			SwitchVRS.init(pGNSSOpt, &m_IGSProducts, true);

			// step 2: 根据基线类型确定RTK模型
			CRTKTCBase* pRTKTC = NULL;
			switch (pGNSSOpt->m_RTKType)
			{
			case 0: pRTKTC = new CRTKTCSBL(); break;
			case 1: pRTKTC = new CRTKTCLBL(); break;
			case 2: pRTKTC = new CRTKTCIBL(); break;
			default: return false;
			}

			// step 3: 准备RTK的数据
			pRTKTC->m_pOBSFFStream[0] = &RoveOBSFFStream;
			pRTKTC->m_pOBSFFStream[1] = &BaseOBSFFStream;
			pRTKTC->m_pEphStore = &m_EphStores;
			pRTKTC->m_pSINSSol = &SINSSol;
			pRTKTC->m_pSINSOpt = &SINSOpt;
			pRTKTC->m_pIGSProducts = &m_IGSProducts;
			pRTKTC->m_pIMUFFStream = &IMUFFStream;
			if (bInitELC) pRTKTC->m_pELCFFStream = &ELCFFStream;
			else          pRTKTC->m_pELCFFStream = NULL;
			if (m_bMultiBaseSite) pGNSSOpt->m_BaseSiteID = BaseOBSFFStream.m_OBSData.BaseSiteID;
			if (pRTKTC->initTC(bReverse) == false) return false;

#ifdef _TwiceProc_LC_ON_

			CIMUFFStream IMUFFStreamTmp = IMUFFStream;
			CSINSSolution SINSSolTmp;
			CLooseCouplingBase* pReverseLCTmp = NULL;
			GPSTIME IMUTimeTmp = IMUFFStreamTmp.m_time;

			if (ls_bCheckScene)
			{
				if (SINSOpt.m_NaviSys == NAVI_E) pReverseLCTmp = new CLooseCouplingECEF();
				else                             pReverseLCTmp = new CLooseCouplingLocal();
				pReverseLCTmp->m_pSINSSol = &SINSSolTmp;
				pReverseLCTmp->m_pIMUFFStream = &IMUFFStreamTmp;
				pReverseLCTmp->m_pELCFFStream = &ELCFFStream;
				pReverseLCTmp->m_pSINSOpt = &SINSOpt;
				pReverseLCTmp->m_pSINSOpt->m_PMode = 2;
				if (pReverseLCTmp->initLC(bReverse) == false) return false;
				pReverseLCTmp->m_pSINSOpt->m_PMode = 3;
			}

			bool flag_IMUTmp = true;

#endif // _TwiceProc_LC_ON_

			// 进度条
			double data_len = fabs(SINSOpt.m_IMUEndTime - SINSOpt.m_IMUBeginTime);
			double data_i = data_len;
			double fProgress = 0.0;
			int iProgress = 0, iProgressPre = 0;
			bool flag_OBS = true;
			bool flag_ELC = true;
			bool flag_IMU = true;
			bool bOBSData = false;
			bool bELCData = false;
			bool bIMUData = false;
			int  errorID = 0;
			GPSTIME gt_OBS;
			GPSTIME gt_ELC;

			while (1)
			{
				if (bInitELC && flag_ELC)
				{
					if (ELCFFStream.getOneRecordData() == false) break;
					gt_ELC = ELCFFStream.m_time;
				}

				if (flag_OBS)
				{
					if (MatchRTKData(BaseOBSFFStream, RoveOBSFFStream, bReverse) == false) break;
					gt_OBS = RoveOBSFFStream.m_time;
				}

				if (flag_IMU)
				{
					if (IMUFFStream.getOneRecordData(true) == false) break;
					IMUTime = IMUFFStream.m_time;
				}

#ifdef _TwiceProc_LC_ON_
				if (ls_bCheckScene)
				{
					if (flag_IMUTmp)
					{
						if (IMUFFStreamTmp.getOneRecordData(true) == false) break;
						IMUTimeTmp = IMUFFStreamTmp.m_time;
					}
				}

				flag_IMUTmp = false;

				if (gt_OBS < IMUTimeTmp)
				{
					bIMUData = true;
					flag_IMUTmp = true;
				}
#endif	

				if (IMUTime < SINSOpt.m_IMUBeginTime) break;

				flag_OBS = flag_ELC = flag_IMU = false;
				bOBSData = bELCData = bIMUData = false;

				if (IMUTime + IMUFFStream.m_DeltT > gt_OBS && gt_OBS >= IMUTime)
				{
					// 执行观测更新,读取OBS+IMU数据
					bOBSData = true;
					flag_OBS = true;
					flag_IMU = true;
				}
				else if (gt_OBS < IMUTime)
				{
					// 执行状态更新,读取IMU数据
					bIMUData = true;
					flag_IMU = true;
				}
				else
				{
					// 不进行任何更新,读取OBS数据
					flag_OBS = true;
				}

				if (bInitELC)
				{
					if (IMUTime + IMUFFStream.m_DeltT > gt_ELC && gt_ELC >= IMUTime)
					{
						// 执行观测更新,读取ELC+IMU数据
						bELCData = true;
						flag_ELC = true;
						flag_IMU = true;
					}
					else if (gt_ELC < IMUTime)
					{
						// 执行状态更新,读取IMU数据
						bIMUData = true;
						flag_IMU = true;
					}
					else
					{
						// 不进行任何更新,读取ELC数据
						flag_ELC = true;
					}
				}

				if (bOBSData || bELCData)
				{
					// 如果同时存在,gt取gt_OBS
					if (bELCData) gt = gt_ELC;
					if (bOBSData) gt = gt_OBS;

					if (gt != IMUTime) IMUFFStream.setMeasUpdate(gt); // 观测更新时设置下进行内插

					// 给VRS赋值并运行
					if (bOBSData && m_bMultiBaseSite)
					{
						pGNSSOpt->m_BaseSiteID = BaseOBSFFStream.m_OBSData.BaseSiteID;
						M31EQU(pGNSSOpt->m_BaseXYZ[pGNSSOpt->m_BaseSiteID], SwitchVRS.m_VRSXYZ);
						SwitchVRS.m_VRSOBSData = BaseOBSFFStream.m_OBSData;
						SwitchVRS.m_EphStoresPostProc = &m_EphStores;
						SwitchVRS.runOneSwitchVRS();
						pRTKTC->m_bSwitchVRS = SwitchVRS.m_bSwitchVRS;
						for (int f = 0; f < NFREQ; f++) pRTKTC->m_VRSAmb[f] = SwitchVRS.m_VRSAmb[f];
					}

#ifdef _TwiceProc_LC_ON_
					if (ls_bCheckScene)
					{
						if (gt != IMUTime) IMUFFStreamTmp.setMeasUpdate(gt); // 观测更新时设置下进行内插

						int Quality = ELCFFStream.m_ELCData.GNSS_DATA.Quality; // 记录原本的Q

						pReverseLCTmp->runOneLC(gt, true);
						if ((fabs(SINSSolTmp.m_MisclosurePos[0]) > 0.2 || fabs(SINSSolTmp.m_MisclosurePos[1]) > 0.2 || fabs(SINSSolTmp.m_MisclosurePos[2]) > 0.2) && (SINSSolTmp.m_IMUFlag&ELCTYPE_POS))
						{
							Quality = 7;
							ls_nCheckPass = 0;
						}
						else if (SINSSolTmp.m_IMUFlag&ELCTYPE_POS)
						{
							ls_nCheckPass++;
						}
						else
						{
							Quality = 7;
						}

						// 检测没问题5个历元内仍不予通过
						if (SINSOpt.m_bTwiceProcLC_NumCheck && ls_nCheckPass < 5) Quality = 7;

						// 满足下列条件认为GNSS没问题，予以通过
						if ((ELCFFStream.m_ELCData.GNSS_DATA.ARRatio > 6 && ELCFFStream.m_ELCData.GNSS_DATA.nSat_ARHold > 6 && ELCFFStream.m_ELCData.GNSS_DATA.AmbState >= ARSTATE_HOLD && MatrixNorm2(3, 1, ELCFFStream.m_ELCData.GNSS_DATA.SepENU) < 0.5) ||
							(ELCFFStream.m_ELCData.GNSS_DATA.ARRatio > 10 && ELCFFStream.m_ELCData.GNSS_DATA.PDOP < 2.0 && MatrixNorm2(3, 1, ELCFFStream.m_ELCData.GNSS_DATA.SepENU) < 0.5))
							Quality = ELCFFStream.m_ELCData.GNSS_DATA.Quality;

						ELCFFStream.m_ELCData.GNSS_DATA.Quality = Quality;
					}
#endif
					errorID = pRTKTC->runOneTC(gt, bOBSData, bELCData);
				}
				else if (bIMUData)
				{
#ifdef _TwiceProc_LC_ON_
					if (ls_bCheckScene) pReverseLCTmp->runOneLC(IMUTimeTmp, false);
#endif
					errorID = pRTKTC->runOneTC(IMUTime, false, false);
				}

				// 观测更新成功,输出有用结果
				if (errorID == 0 && (bOBSData || bELCData))
				{
					if (gs_bIPSAPP || gs_bHCDH || gs_bLJYY) // APP界面信息
					{
						if (gs_bHCDH || gs_bLJYY) data_i = fabs(SINSSol.m_time - SINSOpt.m_IMUEndTime);
						else if (gs_bIPSAPP) data_i = fabs(SINSSol.m_time - SINSOpt.m_IMUBeginTime);
						if (data_i < data_len)
						{
							fProgress = (data_i / data_len) * 100.0; // 进度条
							iProgress = ceil(fProgress);
						}
						SINSSol.FlushAppInfo(fProgress, &m_AppInfo[1], m_SINSSolRpt + 1);
						if (iProgress != iProgressPre)
						{
							DumpProgress(iProgress, m_AppInfo[1]);
						}
						iProgressPre = iProgress;
					}
				}
				else if (errorID == 2) break;

				// 作为下次前向滤波的初始对准结果
				m_SINSInitState.getNxtSINSInitState(SINSSol.m_time, SINSSol.m_XYZPos, SINSSol.m_XYZVel, SINSSol.m_Att, SINSSol.m_GyrosBias, SINSSol.m_AccelBias);
			}

#ifdef _TwiceProc_LC_ON_
			if (pReverseLCTmp) delete pReverseLCTmp;
#endif
			delete pRTKTC;
		}

		if (gs_bUsedPThread) { DUMP_CLOSE_Reverse }
		else { DUMP_CLOSE }

		ReverseFile(m_rtfFile, 2);

		if (SINSOpt.m_bRTS)
		{
			strcpy(m_AppInfo[1].Title, "Reverse RTS: RTK-TCI");
			SINS_RTS_Smoother(true, m_OptFile, m_brtfFile, m_rtsFile, &m_SINSOpt, &m_AppInfo[1].progress);
		}

		m_AppInfo[1].bFinish = true;
		m_AppInfo[1].bValid = false;

		return true;
	}


	bool CSINSApplication::runForwardRTKTCSemi()
	{
		ZeroStruct(m_SINSSolRpt[0], SOLRPT);
		m_AppInfo[0].bFinish = false;
		m_AppInfo[0].bValid = true;
		strcpy(m_AppInfo[0].Title, "Forward KF: RTK-STC");

		if (gs_bUsedPThread) { DUMP_OPEN_Forward(m_ftfmsgFile) }
		else { DUMP_OPEN(m_ftfmsgFile) }

		bool bReverse = false;
		bool bInitELC = false;
		CSINSOption SINSOpt = m_SINSOpt;
		CGNSSOption* pGNSSOpt = &(SINSOpt.m_GNSSOpt);
		pGNSSOpt->m_ResultFilePath = m_ftfFile;
		pGNSSOpt->m_BaseSiteID = 1; // 默认使用第一个基站,未来多基站处理需要改
		if (SINSOpt.m_bIMUProcAll)
		{
			SINSOpt.m_IMUBeginTime = m_IMUDataContainer.m_BeginTime;
			SINSOpt.m_IMUEndTime = m_IMUDataContainer.m_EndTime;
		}

		if (m_ELCDataContainer.m_bValid)
		{
			SINSOpt.m_ELCBeginTime = m_ELCDataContainer.m_BeginTime;
			SINSOpt.m_ELCEndTime = m_ELCDataContainer.m_EndTime;
			bInitELC = true;
		}

		// 只有ELC可用时进行SINS初始化对准,获取初始位置速度和姿态
		if (bInitELC && m_SINSInitState.init(bReverse, &SINSOpt, &m_ELCDataContainer, &m_IMUDataContainer))
		{
			if (!m_SINSInitState.getSINSInitState()) return false;
		}
		else
		{
			string msg("There is no ELC file, fail to get SINS init state!");
			DumpException(msg, LOGT_WARNING | LOGT_AFX, DUMP_LOCATION);
			return false;
		}

		GPSTIME gt, IMUTime;

		// 1. 读取GNSS数据
		COBSFFStream RoveOBSFFStream;
		RoveOBSFFStream.setDataContainer(&m_OBSDataContainers[0]);
		if (RoveOBSFFStream.init(bReverse) == false) return false;
		RoveOBSFFStream.getOneRecordData();
		gt = RoveOBSFFStream.m_time;
		while (RoveOBSFFStream.peakOneRecordTime() <= SINSOpt.m_IMUBeginTime) RoveOBSFFStream.getOneRecordData();
		if (RoveOBSFFStream.CheckDataIsEOF()) return false;

		COBSFFStream BaseOBSFFStream;
		BaseOBSFFStream.setDataContainer(&m_OBSDataContainers[pGNSSOpt->m_BaseSiteID]);
		if (BaseOBSFFStream.init(bReverse) == false) return false;
		BaseOBSFFStream.getOneRecordData();
		gt = BaseOBSFFStream.m_time;
		while (BaseOBSFFStream.peakOneRecordTime() <= SINSOpt.m_IMUBeginTime) BaseOBSFFStream.getOneRecordData();
		if (BaseOBSFFStream.CheckDataIsEOF()) return false;


		// 2. 读取SINS数据
		CIMUFFStream IMUFFStream;
		IMUFFStream.m_IMUDataGapTol = SINSOpt.m_IMUDataGapTol;
		IMUFFStream.setDataContainer(&m_IMUDataContainer);
		if (IMUFFStream.init(bReverse) == false) return false;
		IMUFFStream.getOneRecordData();
		IMUTime = IMUFFStream.m_time;
		while (IMUFFStream.peakOneRecordTime() <= SINSOpt.m_IMUBeginTime) IMUFFStream.getOneRecordData();
		if (IMUFFStream.CheckDataIsEOF()) return false;


		// 3. 读取ELC数据
		CELCFFStream ELCFFStream;
		ELCFFStream.setDataContainer(&m_ELCDataContainer);
		if (ELCFFStream.init(bReverse) == false) return false;
		ELCFFStream.getOneRecordData();
		gt = ELCFFStream.m_time;
		while (ELCFFStream.peakOneRecordTime() <= SINSOpt.m_IMUBeginTime) ELCFFStream.getOneRecordData();
		if (ELCFFStream.CheckDataIsEOF()) return false;


		// 5. 初始化TC
		if (1) // 括号里的类运行完后会自动调用析构函数,关闭相应文件
		{
			CRTKTCSemi RTKTCSemi;

			CSINSSolution SINSSol;
			SINSSol.m_pSINSOpt = &SINSOpt;
			AssignSINSHead(&SINSSol.m_SINSHead);
			if (SINSSol.init(m_ftfFile, true) == false) return false;

			CGNSSSolution GNSSSol;
			GNSSSol.m_pGNSSOpt = m_pGNSSOpt;
			
			///////////////////////////////////////////////////////////////////////////////////////////
			// step 2: 根据基线类型确定RTK模型
			RTKTCSemi.m_pRTK = NULL;
			switch (pGNSSOpt->m_RTKType)
			{
			case 0: RTKTCSemi.m_pRTK = new CRTKPointSBL(); break;
			case 1: RTKTCSemi.m_pRTK = new CRTKPointLBL(); break;
			case 2: RTKTCSemi.m_pRTK = new CRTKPointIBL(); break;
			default: return false;
			}

			// step 3: 准备RTK的数据
			RTKTCSemi.m_pRTK->m_pEphStore = &m_EphStores;
			RTKTCSemi.m_pRTK->m_pGNSSSol = &GNSSSol;
			RTKTCSemi.m_pRTK->m_pGNSSOpt = pGNSSOpt;
			RTKTCSemi.m_pRTK->m_pIGSProducts = &m_IGSProducts;
			RTKTCSemi.m_pRTK->m_pOBSFFStream[0] = &(RoveOBSFFStream);
			RTKTCSemi.m_pRTK->m_pOBSFFStream[1] = &(BaseOBSFFStream);
			
			///////////////////////////////////////////////////////////////////////////////////////////
			// step 4: 准备LC
			RTKTCSemi.m_pLC = NULL;
			if (SINSOpt.m_NaviSys == NAVI_E) RTKTCSemi.m_pLC = new CLooseCouplingECEF();
			else                             RTKTCSemi.m_pLC = new CLooseCouplingLocal();
			RTKTCSemi.m_pLC->m_pSINSSol = &SINSSol;
			RTKTCSemi.m_pLC->m_pIMUFFStream = &IMUFFStream;
			RTKTCSemi.m_pLC->m_pELCFFStream = &ELCFFStream;
			RTKTCSemi.m_pLC->m_pSINSOpt = &SINSOpt;
			if (m_bMultiBaseSite) pGNSSOpt->m_BaseSiteID = BaseOBSFFStream.m_OBSData.BaseSiteID;

			// 初始化半紧组合
			if (RTKTCSemi.initTC(bReverse) == false) return false;

			///////////////////////////////////////////////////////////////////////////////////////////
			// step 5: 运行
			// 进度条
			double data_len = fabs(SINSOpt.m_IMUEndTime - SINSOpt.m_IMUBeginTime);
			double data_i = 0;
			double fProgress = 0.0;
			int iProgress = 0, iProgressPre = 0;
			bool flag_OBS = true;
			bool flag_ELC = true;
			bool flag_IMU = true;
			bool bOBSData = false;
			bool bELCData = false;
			bool bIMUData = false;
			int  errorID = 0;
			GPSTIME gt_OBS;
			GPSTIME gt_ELC;

			while (1)
			{
				if (bInitELC && flag_ELC)
				{
					if (ELCFFStream.getOneRecordData() == false) break;
					gt_ELC = ELCFFStream.m_time;
				}

				if (flag_OBS)
				{
					if (MatchRTKData(BaseOBSFFStream, RoveOBSFFStream, bReverse) == false) break;
					gt_OBS = RoveOBSFFStream.m_time;
				}

				if (flag_IMU)
				{
					if (IMUFFStream.getOneRecordData(true) == false) break;
					IMUTime = IMUFFStream.m_time;
				}

				if (IMUTime > SINSOpt.m_IMUEndTime) break;

				flag_OBS = flag_ELC = flag_IMU = false;
				bOBSData = bELCData = bELCData = false;

				if (IMUTime - IMUFFStream.m_DeltT < gt_OBS && gt_OBS <= IMUTime)
				{
					// 执行观测更新,读取OBS+IMU数据
					bOBSData = true;
					flag_OBS = true;
					flag_IMU = true;
				}
				else if (gt_OBS > IMUTime)
				{
					// 执行状态更新,读取IMU数据
					bIMUData = true;
					flag_IMU = true;
				}
				else
				{
					// 不进行任何更新,读取OBS数据
					flag_OBS = true;
				}

				if (bInitELC)
				{
					if (IMUTime - IMUFFStream.m_DeltT < gt_ELC && gt_ELC <= IMUTime)
					{
						// 执行观测更新,读取ELC+IMU数据
						bELCData = true;
						flag_ELC = true;
						flag_IMU = true;
					}
					else if (gt_ELC > IMUTime)
					{
						// 执行状态更新,读取IMU数据
						bIMUData = true;
						flag_IMU = true;
					}
					else
					{
						// 不进行任何更新,读取ELC数据
						flag_ELC = true;
					}
				}

				if (bOBSData || bELCData)
				{
					// 如果同时存在,gt取gt_OBS
					if (bELCData) gt = gt_ELC;
					if (bOBSData) gt = gt_OBS;

					if (gt != IMUTime) IMUFFStream.setMeasUpdate(gt); // 观测更新时设置下进行内插
					errorID = RTKTCSemi.runOneTC(gt, bOBSData, bELCData);
				}
				else if (bIMUData)
				{
					errorID = RTKTCSemi.runOneTC(IMUTime, false, false);
				}

				// 观测更新成功,输出有用结果
				if (errorID == 0 && (bOBSData || bELCData))
				{
					if (gs_bIPSAPP || gs_bHCDH || gs_bLJYY) // APP界面信息
					{
						data_i = fabs(SINSSol.m_time - SINSOpt.m_IMUBeginTime);
						if (data_i < data_len)
						{
							fProgress = (data_i / data_len) * 100.0; // 进度条
							iProgress = ceil(fProgress);
						}
						SINSSol.FlushAppInfo(fProgress, &m_AppInfo[0], m_SINSSolRpt);
						if (iProgress != iProgressPre)
						{
							DumpProgress(iProgress, m_AppInfo[0]);
						}
						iProgressPre = iProgress;
					}
				}
				else if (errorID == 2) break;
			}

			delete RTKTCSemi.m_pRTK;
			delete RTKTCSemi.m_pLC;

			// 作为下次后向滤波的初始对准结果
			m_SINSInitState.getNxtSINSInitState(SINSSol.m_time, SINSSol.m_XYZPos, SINSSol.m_XYZVel, SINSSol.m_Att, SINSSol.m_GyrosBias, SINSSol.m_AccelBias);
		}

		if (gs_bUsedPThread) { DUMP_CLOSE_Forward }
		else { DUMP_CLOSE }

		if (SINSOpt.m_bRTS)
		{
			strcpy(m_AppInfo[0].Title, "Forward RTS: RTK-STC");
			SINS_RTS_Smoother(false, m_OptFile, m_bftfFile, m_ftsFile, &m_SINSOpt, &m_AppInfo[0].progress);
		}

		m_AppInfo[0].bFinish = true;
		m_AppInfo[0].bValid = false;

		return true;
	}


	bool CSINSApplication::runReverseRTKTCSemi()
	{
		ZeroStruct(m_SINSSolRpt[1], SOLRPT);
		m_AppInfo[1].bFinish = false;
		m_AppInfo[1].bValid = true;
		strcpy(m_AppInfo[1].Title, "Reverse KF: RTK-STC");

		if (gs_bUsedPThread) { DUMP_OPEN_Reverse(m_rtfmsgFile) }
		else { DUMP_OPEN(m_rtfmsgFile) }

		bool bReverse = true;
		bool bInitELC = false;
		CSINSOption SINSOpt = m_SINSOpt;
		CGNSSOption* pGNSSOpt = &(SINSOpt.m_GNSSOpt);
		pGNSSOpt->m_ResultFilePath = m_rtfFile;
		pGNSSOpt->m_BaseSiteID = 1; // 默认使用第一个基站,未来多基站处理需要改
		if (SINSOpt.m_bIMUProcAll)
		{
			SINSOpt.m_IMUBeginTime = m_IMUDataContainer.m_BeginTime;
			SINSOpt.m_IMUEndTime = m_IMUDataContainer.m_EndTime;
		}

		if (m_ELCDataContainer.m_bValid)
		{
			SINSOpt.m_ELCBeginTime = m_ELCDataContainer.m_BeginTime;
			SINSOpt.m_ELCEndTime = m_ELCDataContainer.m_EndTime;
			bInitELC = true;
		}

		// 只有ELC可用时进行SINS初始化对准,获取初始位置速度和姿态
		if (bInitELC && m_SINSInitState.init(bReverse, &SINSOpt, &m_ELCDataContainer, &m_IMUDataContainer))
		{
			if (!m_SINSInitState.getSINSInitState()) return false;
		}
		else
		{
			string msg("There is no ELC file, fail to get SINS init state!");
			DumpException(msg, LOGT_WARNING | LOGT_AFX, DUMP_LOCATION);
			return false;
		}

		GPSTIME gt, IMUTime;

		// 1. 读取GNSS数据
		COBSFFStream RoveOBSFFStream;
		RoveOBSFFStream.setDataContainer(&m_OBSDataContainers[0]);
		if (RoveOBSFFStream.init(bReverse) == false) return false;
		RoveOBSFFStream.getOneRecordData();
		gt = RoveOBSFFStream.m_time;
		while (RoveOBSFFStream.peakOneRecordTime() >= SINSOpt.m_IMUEndTime) RoveOBSFFStream.getOneRecordData();
		if (RoveOBSFFStream.CheckDataIsEOF()) return false;

		COBSFFStream BaseOBSFFStream;
		BaseOBSFFStream.setDataContainer(&m_OBSDataContainers[pGNSSOpt->m_BaseSiteID]);
		if (BaseOBSFFStream.init(bReverse) == false) return false;
		BaseOBSFFStream.getOneRecordData();
		gt = BaseOBSFFStream.m_time;
		while (BaseOBSFFStream.peakOneRecordTime() >= SINSOpt.m_IMUEndTime) BaseOBSFFStream.getOneRecordData();
		if (BaseOBSFFStream.CheckDataIsEOF()) return false;


		// 2. 读取SINS数据
		CIMUFFStream IMUFFStream;
		IMUFFStream.m_IMUDataGapTol = SINSOpt.m_IMUDataGapTol;
		IMUFFStream.setDataContainer(&m_IMUDataContainer);
		if (IMUFFStream.init(bReverse) == false) return false;
		IMUFFStream.getOneRecordData();
		IMUTime = IMUFFStream.m_time;
		while (IMUFFStream.peakOneRecordTime() >= SINSOpt.m_IMUEndTime) IMUFFStream.getOneRecordData();
		if (IMUFFStream.CheckDataIsEOF()) return false;

		// 只是为了与原来的对齐,可以去掉???
		// IMUFFStream.getOneRecordData();

		// 3. 读取ELC数据
		CELCFFStream ELCFFStream;
		ELCFFStream.setDataContainer(&m_ELCDataContainer);
		if (ELCFFStream.init(bReverse) == false) return false;
		ELCFFStream.getOneRecordData();
		gt = ELCFFStream.m_time;
		while (ELCFFStream.peakOneRecordTime() >= SINSOpt.m_IMUEndTime) ELCFFStream.getOneRecordData();
		if (ELCFFStream.CheckDataIsEOF()) return false;


		// 4. 初始化TC
		if (1) // 括号里的类运行完后会自动调用析构函数,关闭相应文件
		{
			CRTKTCSemi RTKTCSemi;

			CSINSSolution SINSSol;
			SINSSol.m_pSINSOpt = &SINSOpt;
			AssignSINSHead(&SINSSol.m_SINSHead);
			if (SINSSol.init(m_rtfFile, true) == false) return false;

			CGNSSSolution GNSSSol;
			GNSSSol.m_pGNSSOpt = m_pGNSSOpt;

			///////////////////////////////////////////////////////////////////////////////////////////
			
			// step 2: 根据基线类型确定RTK模型
			RTKTCSemi.m_pRTK = NULL;
			switch (pGNSSOpt->m_RTKType)
			{
			case 0: RTKTCSemi.m_pRTK = new CRTKPointSBL(); break;
			case 1: RTKTCSemi.m_pRTK = new CRTKPointLBL(); break;
			case 2: RTKTCSemi.m_pRTK = new CRTKPointIBL(); break;
			default: return false;
			}

			// step 3: 准备RTK的数据
			RTKTCSemi.m_pRTK->m_pEphStore = &m_EphStores;
			RTKTCSemi.m_pRTK->m_pGNSSSol = &GNSSSol;
			RTKTCSemi.m_pRTK->m_pGNSSOpt = pGNSSOpt;
			RTKTCSemi.m_pRTK->m_pIGSProducts = &m_IGSProducts;
			RTKTCSemi.m_pRTK->m_pOBSFFStream[0] = &(RoveOBSFFStream);
			RTKTCSemi.m_pRTK->m_pOBSFFStream[1] = &(BaseOBSFFStream);

			///////////////////////////////////////////////////////////////////////////////////////////
			// step 4: 准备LC
			RTKTCSemi.m_pLC = NULL;
			if (SINSOpt.m_NaviSys == NAVI_E) RTKTCSemi.m_pLC = new CLooseCouplingECEF();
			else                             RTKTCSemi.m_pLC = new CLooseCouplingLocal();
			RTKTCSemi.m_pLC->m_pSINSSol = &SINSSol;
			RTKTCSemi.m_pLC->m_pIMUFFStream = &IMUFFStream;
			RTKTCSemi.m_pLC->m_pELCFFStream = &ELCFFStream;
			RTKTCSemi.m_pLC->m_pSINSOpt = &SINSOpt;
			if (m_bMultiBaseSite) pGNSSOpt->m_BaseSiteID = BaseOBSFFStream.m_OBSData.BaseSiteID;

			// 初始化半紧组合
			if (RTKTCSemi.initTC(bReverse) == false) return false;

			// 进度条
			double data_len = fabs(SINSOpt.m_IMUEndTime - SINSOpt.m_IMUBeginTime);
			double data_i = data_len;
			double fProgress = 0.0;
			int iProgress = 0, iProgressPre = 0;
			bool flag_OBS = true;
			bool flag_ELC = true;
			bool flag_IMU = true;
			bool bOBSData = false;
			bool bELCData = false;
			bool bIMUData = false;
			int  errorID = 0;
			GPSTIME gt_OBS;
			GPSTIME gt_ELC;

			while (1)
			{
				if (bInitELC && flag_ELC)
				{
					if (ELCFFStream.getOneRecordData() == false) break;
					gt_ELC = ELCFFStream.m_time;
				}

				if (flag_OBS)
				{
					if (MatchRTKData(BaseOBSFFStream, RoveOBSFFStream, bReverse) == false) break;
					gt_OBS = RoveOBSFFStream.m_time;
				}

				if (flag_IMU)
				{
					if (IMUFFStream.getOneRecordData(true) == false) break;
					IMUTime = IMUFFStream.m_time;
				}

				if (IMUTime < SINSOpt.m_IMUBeginTime) break;

				flag_OBS = flag_ELC = flag_IMU = false;
				bOBSData = bELCData = bELCData = false;

				if (IMUTime + IMUFFStream.m_DeltT > gt_OBS && gt_OBS >= IMUTime)
				{
					// 执行观测更新,读取OBS+IMU数据
					bOBSData = true;
					flag_OBS = true;
					flag_IMU = true;
				}
				else if (gt_OBS < IMUTime)
				{
					// 执行状态更新,读取IMU数据
					bIMUData = true;
					flag_IMU = true;
				}
				else
				{
					// 不进行任何更新,读取OBS数据
					flag_OBS = true;
				}

				if (bInitELC)
				{
					if (IMUTime + IMUFFStream.m_DeltT > gt_ELC && gt_ELC >= IMUTime)
					{
						// 执行观测更新,读取ELC+IMU数据
						bELCData = true;
						flag_ELC = true;
						flag_IMU = true;
					}
					else if (gt_ELC < IMUTime)
					{
						// 执行状态更新,读取IMU数据
						bIMUData = true;
						flag_IMU = true;
					}
					else
					{
						// 不进行任何更新,读取ELC数据
						flag_ELC = true;
					}
				}

				if (bOBSData || bELCData)
				{
					// 如果同时存在,gt取gt_OBS
					if (bELCData) gt = gt_ELC;
					if (bOBSData) gt = gt_OBS;

					if (gt != IMUTime) IMUFFStream.setMeasUpdate(gt); // 观测更新时设置下进行内插
					errorID = RTKTCSemi.runOneTC(gt, bOBSData, bELCData);
				}
				else if (bIMUData)
				{
					errorID = RTKTCSemi.runOneTC(IMUTime, false, false);
				}

				// 观测更新成功,输出有用结果
				if (errorID == 0 && (bOBSData || bELCData))
				{
					if (gs_bIPSAPP || gs_bHCDH || gs_bLJYY) // APP界面信息
					{
						if (gs_bHCDH || gs_bLJYY) data_i = fabs(SINSSol.m_time - SINSOpt.m_IMUEndTime);
						else if (gs_bIPSAPP) data_i = fabs(SINSSol.m_time - SINSOpt.m_IMUBeginTime);
						if (data_i < data_len)
						{
							fProgress = (data_i / data_len) * 100.0; // 进度条
							iProgress = ceil(fProgress);
						}
						SINSSol.FlushAppInfo(fProgress, &m_AppInfo[1], m_SINSSolRpt + 1);
						if (iProgress != iProgressPre)
						{
							DumpProgress(iProgress, m_AppInfo[1]);
						}
						iProgressPre = iProgress;
					}
				}
				else if (errorID == 2) break;
			}

			delete RTKTCSemi.m_pRTK;
			delete RTKTCSemi.m_pLC;

			// 作为下次前向滤波的初始对准结果
			m_SINSInitState.getNxtSINSInitState(SINSSol.m_time, SINSSol.m_XYZPos, SINSSol.m_XYZVel, SINSSol.m_Att, SINSSol.m_GyrosBias, SINSSol.m_AccelBias);
		}

		if (gs_bUsedPThread) { DUMP_CLOSE_Reverse }
		else { DUMP_CLOSE }

		ReverseFile(m_rtfFile, 2);

		if (SINSOpt.m_bRTS)
		{
			strcpy(m_AppInfo[1].Title, "Reverse RTS: RTK-STC");
			SINS_RTS_Smoother(true, m_OptFile, m_brtfFile, m_rtsFile, &m_SINSOpt, &m_AppInfo[1].progress);
		}

		m_AppInfo[1].bFinish = true;
		m_AppInfo[1].bValid = false;

		return true;
	}


	bool CSINSApplication::runTC()
	{
//#ifdef _TwiceProc_LC_ON_
		string filename = m_pGNSSOpt->m_ProjPath + FILEPATHSEP + m_pGNSSOpt->m_ProjName + ".rpt";
		ls_bCheckScene = CheckScene(filename);
		if (gs_bHCDH || gs_bLJYY) ls_bCheckScene = true;

		// 如果惯导是低成本的，不要开启
		if (ls_bCheckScene && (m_SINSOpt.m_SSP.PNSD_AccelBiasX > 1e-5 || m_SINSOpt.m_SSP.PNSD_AccelBiasY > 1e-5 || m_SINSOpt.m_SSP.PNSD_AccelBiasZ > 1e-5)
			&& (m_SINSOpt.m_SSP.PNSD_GyrosBiasX > 1e-5 * D2R || m_SINSOpt.m_SSP.PNSD_GyrosBiasY > 1e-5 * D2R || m_SINSOpt.m_SSP.PNSD_GyrosBiasZ > 1e-5 * D2R))
			ls_bCheckScene = false;

//#endif // _TwiceProc_LC_ON_

		if (m_SINSOpt.m_FilterType == 5)
		{
			if (m_pGNSSOpt->m_PMode > GNSSMODE_SINGLE_MAX) runRTKTCPostIAR();
			else                                        runPPPTCPostIAR();
			return true;
		}


		if(m_SINSOpt.m_FilterType == 0 || m_SINSOpt.m_FilterType == 2)
		{
			m_AppInfo[0].FilterType = 0;
			if (gs_bUsedPThread && m_SINSOpt.m_FilterType == 2)
			{
				if (m_pGNSSOpt->m_PMode > GNSSMODE_SINGLE_MAX)
				{
					if (m_SINSOpt.m_PMode == 5) gs_Forward_Thread = new thread(&sinstk::CSINSApplication::runForwardRTKTCSemi, this);
					else                        gs_Forward_Thread = new thread(&sinstk::CSINSApplication::runForwardRTKTC, this);
				}
				else
				{
					gs_Forward_Thread = new thread(&sinstk::CSINSApplication::runForwardPPPTC, this);
				}
			}
			else
			{
				if (m_pGNSSOpt->m_PMode > GNSSMODE_SINGLE_MAX)
				{
					if(m_SINSOpt.m_PMode == 5) runForwardRTKTCSemi();
					else                       runForwardRTKTC();
				}
				else
				{
					runForwardPPPTC();
				}
			}
		}

		if (gs_bHCDH || gs_bLJYY)
		{
			if (m_SINSInitDMIModel.init(&m_SINSOpt, m_bftsFile, m_DMIFile)) m_SINSInitDMIModel.runInstallEstimate();
		}

		if(m_SINSOpt.m_FilterType == 1 || m_SINSOpt.m_FilterType == 2)
		{
			m_AppInfo[1].FilterType = 1;
			if (gs_bUsedPThread && m_SINSOpt.m_FilterType == 2)
			{
				if (m_pGNSSOpt->m_PMode > GNSSMODE_SINGLE_MAX)
				{
					if (m_SINSOpt.m_PMode == 5) gs_Reverse_Thread = new thread(&sinstk::CSINSApplication::runReverseRTKTCSemi, this);
					else                        gs_Reverse_Thread = new thread(&sinstk::CSINSApplication::runReverseRTKTC, this);
				}
				else
				{
					gs_Reverse_Thread = new thread(&sinstk::CSINSApplication::runReversePPPTC, this);
				}
			}
			else
			{
				if (m_pGNSSOpt->m_PMode > GNSSMODE_SINGLE_MAX)
				{
					if (m_SINSOpt.m_PMode == 5) runReverseRTKTCSemi();
					else                        runReverseRTKTC();
				}
				else
				{
					runReversePPPTC();
				}
			}
		}
		
		// 再做一遍前向
		if ((gs_bHCDH || gs_bLJYY) && m_SINSOpt.m_FilterType == 2)
		{
			m_AppInfo[0].FilterType = 2;
			if (gs_bUsedPThread)
			{
				if (m_pGNSSOpt->m_PMode > GNSSMODE_SINGLE_MAX)
				{
					if (m_SINSOpt.m_PMode == 5) gs_Forward_Thread = new thread(&sinstk::CSINSApplication::runForwardRTKTCSemi, this);
					else                        gs_Forward_Thread = new thread(&sinstk::CSINSApplication::runForwardRTKTC, this);
				}
				else
				{
					gs_Forward_Thread = new thread(&sinstk::CSINSApplication::runForwardPPPTC, this);
				}
			}
			else
			{
				if (m_pGNSSOpt->m_PMode > GNSSMODE_SINGLE_MAX)
				{
					if (m_SINSOpt.m_PMode == 5) runForwardRTKTCSemi();
					else                        runForwardRTKTC();
				}
				else
				{
					runForwardPPPTC();
				}
			}
		}

		if (m_SINSOpt.m_FilterType == 2)
		{
			// 等待线程结束,再往下继续运行
			if (gs_bUsedPThread)
			{
				gs_Forward_Thread->join();
				gs_Reverse_Thread->join();

				delete gs_Forward_Thread; gs_Forward_Thread = NULL;
				delete gs_Reverse_Thread; gs_Reverse_Thread = NULL;
			}

			m_AppInfo[0].bFinish = false;
			m_AppInfo[0].bValid = true;
			strcpy(m_AppInfo[0].Title, "Combination: TCI");
			if (m_SINSOpt.m_bRTS)
			SINS_Combination(m_OptFile, m_bftsFile, m_brtsFile, m_dtsFile, &m_SINSOpt, &m_AppInfo[0].progress);
			SINS_Combination(m_OptFile, m_bftfFile, m_brtfFile, m_dtfFile, &m_SINSOpt, &m_AppInfo[0].progress); // 无RTS的双向组合平滑
			m_AppInfo[0].bFinish = true;
			m_AppInfo[0].bValid = false;
		}

		if (m_SINSOpt.m_ExpXMLFile != "")
		{
			char cFileTitle[1024] = "";
			getFileInfo(_C(m_IMUFile.c_str()), cFileTitle, NULL, NULL, NULL);
			string ExportFile = m_FilePath + FILEPATHSEP + cFileTitle + ".PosT";

			string SatFile = "";
			if (gs_bHCDH)
			{
				getFileInfo(_C(m_OBSFile[0].c_str()), cFileTitle, NULL, NULL, NULL);
				SatFile = m_SINSOpt.m_GNSSOpt.m_GNSS_ResultPath + "R_RTK" + FILEPATHSEP + cFileTitle + "." + gs_GNSSFileTypes[GFT_FFP] + "_sat2";
			}

			CSINSSolutionExport SINSExportTest;
			SINSExportTest.initSINSExp(m_bdtsFile, m_IMUFile, m_SINSOpt.m_MarkFile, m_SINSOpt.m_ExpXMLFile, ExportFile, m_SINSOpt.m_RotAngleIMU, SatFile);
			SINSExportTest.ExportSolution();
		}

		return true;
	}


	bool CSINSApplication::runINS()
	{
		if (m_SINSOpt.m_FilterType == 0 || m_SINSOpt.m_FilterType == 2)
		{
			ZeroStruct(m_SINSSolRpt[0], SOLRPT);
			m_AppInfo[0].bFinish = false;
			m_AppInfo[0].bValid = false;
			strcpy(m_AppInfo[0].Title, "Forward : INS");
			if (gs_bUsedPThread) { DUMP_OPEN_Forward(m_flfmsgFile) }
			else { DUMP_OPEN(m_flfmsgFile) }

			bool bReverse = false;
			CSINSOption SINSOpt = m_SINSOpt;

			// 2. 读取SINS数据
			CIMUFFStream IMUFFStream;
			IMUFFStream.m_IMUDataGapTol = SINSOpt.m_IMUDataGapTol;
			IMUFFStream.setDataContainer(&m_IMUDataContainer);
			if (IMUFFStream.init(bReverse) == false) return false;
			IMUFFStream.getOneRecordData();
			GPSTIME IMUTime = IMUFFStream.m_time;
			if (SINSOpt.m_bIMUProcAll)
			{
				SINSOpt.m_IMUBeginTime = IMUFFStream.m_BeginTime;
				SINSOpt.m_IMUEndTime = IMUFFStream.m_EndTime;
			}
			else
			{
				if (SINSOpt.m_IMUBeginTime < IMUFFStream.m_BeginTime) SINSOpt.m_IMUBeginTime = IMUFFStream.m_BeginTime;
				if (SINSOpt.m_IMUEndTime   > IMUFFStream.m_EndTime)   SINSOpt.m_IMUEndTime = IMUFFStream.m_EndTime;
				while (IMUTime < SINSOpt.m_IMUBeginTime && IMUFFStream.getOneRecordData()) IMUTime = IMUFFStream.m_time;
			}
			if (IMUFFStream.CheckDataIsEOF()) return false;

			char msg[1024] = { '\0' };
			sprintf(msg, "INS initialization finished time: %d %.4f.", SINSOpt.m_IMUBeginTime.GPSWeek, SINSOpt.m_IMUBeginTime.secsOfWeek + SINSOpt.m_IMUBeginTime.fracOfSec);
			DumpException(string(msg), LOGT_INFO, DUMP_LOCATION);
			sprintf(msg, "Set init pos: %13.4f %13.4f %13.4f.", SINSOpt.m_InitPosf[0], SINSOpt.m_InitPosf[1], SINSOpt.m_InitPosf[2]);
			DumpException(string(msg), LOGT_INFO, DUMP_LOCATION);
			sprintf(msg, "Set init vel: %13.4f %13.4f %13.4f.", SINSOpt.m_InitVelf[0], SINSOpt.m_InitVelf[1], SINSOpt.m_InitVelf[2]);
			DumpException(string(msg), LOGT_INFO, DUMP_LOCATION);
			sprintf(msg, "Set init att: %13.4f %13.4f %13.4f.", SINSOpt.m_InitAttf[0] * R2D, SINSOpt.m_InitAttf[1] * R2D, SINSOpt.m_InitAttf[2] * R2D);
			DumpException(string(msg), LOGT_INFO, DUMP_LOCATION);

			// 4. 初始化INS
			if (1) // 括号里的类运行完后会自动调用析构函数,关闭相应文件
			{
				CSINSSolution SINSSol;
				SINSSol.m_pSINSOpt = &SINSOpt;
				AssignSINSHead(&SINSSol.m_SINSHead);
				if (SINSSol.init(m_flfFile, true, false) == false) return false;

				CLooseCouplingBase* pForwardLC;
				if (SINSOpt.m_NaviSys == NAVI_E) pForwardLC = new CLooseCouplingECEF();
				else                             pForwardLC = new CLooseCouplingLocal();
				pForwardLC->m_pSINSSol = &SINSSol;
				pForwardLC->m_pIMUFFStream = &IMUFFStream;
				pForwardLC->m_pELCFFStream = NULL;
				pForwardLC->m_pSINSOpt = &SINSOpt;
				if (pForwardLC->initLC(bReverse) == false) return false;

				// 进度条
				double data_len = fabs(SINSOpt.m_IMUEndTime - SINSOpt.m_IMUBeginTime);
				double data_i = 0;
				double fProgress = 0.0;
				int iProgress = 0, iProgressPre = 0;
				bool flag_ELC = true;
				bool flag_IMU = true;
				bool bELCData = true;
				bool bIMUData = true;
				int  errorID = 0;

				GPSTIME gt(SINSOpt.m_IMUBeginTime.GPSWeek, SINSOpt.m_IMUBeginTime.secsOfWeek);

				while (1)
				{
					if (flag_ELC)
					{
						gt = gt + 1.0;
					}

					if (flag_IMU)
					{
						if (IMUFFStream.getOneRecordData(true) == false) break;
						IMUTime = IMUFFStream.m_time;
					}

					if (IMUTime > SINSOpt.m_IMUEndTime) break;

					flag_ELC = flag_IMU = bELCData = bIMUData = false;

					// 1. 判断LC的操作
					if (IMUTime - IMUFFStream.m_DeltT < gt && gt <= IMUTime)
					{
						// 执行观测更新,读取两者数据
						bELCData = true;
						flag_ELC = true;
						flag_IMU = true;
					}
					else if (gt > IMUTime)
					{
						// 执行状态更新,读取IMU数据
						bIMUData = true;
						flag_IMU = true;
					}
					else
					{
						// 不进行任何更新,读取ELC数据
						flag_ELC = true;
					}

					if (bELCData)
					{
						if (gt != IMUTime) IMUFFStream.setMeasUpdate(gt); // 观测更新时设置下进行内插
						errorID = pForwardLC->runOneLC(gt, false);
					}
					else if (bIMUData)
					{
						errorID = pForwardLC->runOneLC(IMUTime, false);
					}

					// 观测更新成功,输出有用结果
					if (errorID == 0 && bELCData)
					{
						if (gs_bIPSAPP) // APP界面信息
						{
							fProgress = ((++data_i) / data_len) * 100.0; // 进度条
							iProgress = ceil(fProgress);
							SINSSol.FlushAppInfo(fProgress, &m_AppInfo[0], m_SINSSolRpt);
							if (iProgress != iProgressPre)
							{
								DumpProgress(iProgress, m_AppInfo[0]);
							}
							iProgressPre = iProgress;
						}
					}
					else if (errorID == 2) break;
				}

				// 删除,防止内存泄露,不删除将报错
				delete pForwardLC;
			}

			if (gs_bUsedPThread) { DUMP_CLOSE_Forward }
			else { DUMP_CLOSE }

			m_AppInfo[0].bFinish = true;
			m_AppInfo[0].bValid = false;

			return true;
		}
		else
		{
			ZeroStruct(m_SINSSolRpt[1], SOLRPT);
			m_AppInfo[1].bFinish = false;
			m_AppInfo[1].bValid = false;
			strcpy(m_AppInfo[1].Title, "Reverse : INS");
			if (gs_bUsedPThread) { DUMP_OPEN_Reverse(m_rlfmsgFile) }
			else { DUMP_OPEN(m_rlfmsgFile) }

			bool bReverse = true;
			CSINSOption SINSOpt = m_SINSOpt;

			// 2. 读取SINS数据
			CIMUFFStream IMUFFStream;
			IMUFFStream.m_IMUDataGapTol = SINSOpt.m_IMUDataGapTol;
			IMUFFStream.setDataContainer(&m_IMUDataContainer);
			if (IMUFFStream.init(bReverse) == false) return false;
			IMUFFStream.getOneRecordData();
			GPSTIME IMUTime = IMUFFStream.m_time;
			if (SINSOpt.m_bIMUProcAll)
			{
				SINSOpt.m_IMUBeginTime = IMUFFStream.m_BeginTime;
				SINSOpt.m_IMUEndTime = IMUFFStream.m_EndTime;
			}
			else
			{
				if (SINSOpt.m_IMUBeginTime < IMUFFStream.m_BeginTime) SINSOpt.m_IMUBeginTime = IMUFFStream.m_BeginTime;
				if (SINSOpt.m_IMUEndTime   > IMUFFStream.m_EndTime)   SINSOpt.m_IMUEndTime = IMUFFStream.m_EndTime;
				while (IMUTime > SINSOpt.m_IMUEndTime && IMUFFStream.getOneRecordData()) IMUTime = IMUFFStream.m_time;
			}
			if (IMUFFStream.CheckDataIsEOF()) return false;

			// 只是为了与原来的对齐,可以去掉
			IMUFFStream.getOneRecordData();

			char msg[1024] = { '\0' };
			sprintf(msg, "INS initialization finished time: %d %.4f.", SINSOpt.m_IMUEndTime.GPSWeek, SINSOpt.m_IMUEndTime.secsOfWeek + SINSOpt.m_IMUEndTime.fracOfSec);
			DumpException(string(msg), LOGT_INFO, DUMP_LOCATION);
			sprintf(msg, "Set init pos: %13.4f %13.4f %13.4f.", SINSOpt.m_InitPosb[0], SINSOpt.m_InitPosb[1], SINSOpt.m_InitPosb[2]);
			DumpException(string(msg), LOGT_INFO, DUMP_LOCATION);
			sprintf(msg, "Set init vel: %13.4f %13.4f %13.4f.", SINSOpt.m_InitVelb[0], SINSOpt.m_InitVelb[1], SINSOpt.m_InitVelb[2]);
			DumpException(string(msg), LOGT_INFO, DUMP_LOCATION);
			sprintf(msg, "Set init att: %13.4f %13.4f %13.4f.", SINSOpt.m_InitAttb[0] * R2D, SINSOpt.m_InitAttb[1] * R2D, SINSOpt.m_InitAttb[2] * R2D);
			DumpException(string(msg), LOGT_INFO, DUMP_LOCATION);

			// 4. 初始化LC
			if (1) // 括号里的类运行完后会自动调用析构函数,关闭相应文件
			{
				CSINSSolution SINSSol;
				SINSSol.m_pSINSOpt = &SINSOpt;
				AssignSINSHead(&SINSSol.m_SINSHead);
				if (SINSSol.init(m_rlfFile, true, false) == false) return false;

				CLooseCouplingBase* pReverseLC;
				if (SINSOpt.m_NaviSys == NAVI_E) pReverseLC = new CLooseCouplingECEF();
				else                             pReverseLC = new CLooseCouplingLocal();
				pReverseLC->m_pSINSSol = &SINSSol;
				pReverseLC->m_pIMUFFStream = &IMUFFStream;
				pReverseLC->m_pELCFFStream = NULL;
				pReverseLC->m_pSINSOpt = &SINSOpt;
				if (pReverseLC->initLC(bReverse) == false) return false;

				// 进度条
				double data_len = fabs(SINSOpt.m_IMUEndTime - SINSOpt.m_IMUBeginTime);
				double data_i = 0;
				double fProgress = 0.0;
				int iProgress = 0, iProgressPre = 0;
				bool flag_ELC = true;
				bool flag_IMU = true;
				bool bELCData = true;
				bool bIMUData = true;
				int  errorID = 0;

				GPSTIME gt(SINSOpt.m_IMUEndTime.GPSWeek, SINSOpt.m_IMUEndTime.secsOfWeek);
				gt = gt + 1;

				while (1)
				{
					if (flag_ELC)
					{
						gt = gt - 1;
					}

					if (flag_IMU)
					{
						if (IMUFFStream.getOneRecordData(true) == false) break;
						IMUTime = IMUFFStream.m_time;
					}

					if (IMUTime < SINSOpt.m_IMUBeginTime) break;

					flag_ELC = flag_IMU = bELCData = bIMUData = false;

					// 1. 判断LC的操作
					if (IMUTime + IMUFFStream.m_DeltT > gt && gt >= IMUTime)
					{
						// 执行观测更新,读取两者数据
						bELCData = true;
						flag_ELC = true;
						flag_IMU = true;
					}
					else if (gt < IMUTime)
					{
						// 执行状态更新,读取IMU数据
						bIMUData = true;
						flag_IMU = true;
					}
					else
					{
						// 不进行任何更新,读取ELC数据
						flag_ELC = true;
					}

					if (bELCData)
					{
						if (gt != IMUTime) IMUFFStream.setMeasUpdate(gt); // 观测更新时设置下进行内插
						errorID = pReverseLC->runOneLC(gt, false);
					}
					else if (bIMUData)
					{
						errorID = pReverseLC->runOneLC(IMUTime, false);
					}

					// 观测更新成功,输出有用结果
					if (errorID == 0 && bELCData)
					{
						if (gs_bIPSAPP) // APP界面信息
						{
							fProgress = ((++data_i) / data_len) * 100.0; // 进度条
							iProgress = ceil(fProgress);
							SINSSol.FlushAppInfo(fProgress, &m_AppInfo[0], m_SINSSolRpt);
							if (iProgress != iProgressPre)
							{
								DumpProgress(iProgress, m_AppInfo[0]);
							}
							iProgressPre = iProgress;
						}
					}
					else if (errorID == 2) break;
				}

				// 删除,防止内存泄露,不删除将报错
				delete pReverseLC;
			}

			if (gs_bUsedPThread) { DUMP_CLOSE_Reverse }
			else { DUMP_CLOSE }

			ReverseFile(m_rlfFile, 2);

			m_AppInfo[1].bFinish = true;
			m_AppInfo[1].bValid = false;
		}

		return true;
	}


	bool CSINSApplication::GNSSIntegrityCheck()
	{
		// 读取ELC，如果固定解连33%都不到，关闭连续验后设置
		CGNSSSolution dfpSol;
		if (dfpSol.init(m_GNSSFile, false, false) == false) return false;
		double AllSpan = 0.0, PassSpan = 0.0, dt = 0.0;
		GPSTIME TimePre(-1, 0);
		while (dfpSol.readSolution())
		{
			if (TimePre.GPSWeek != -1)
			{
				dt = dfpSol.m_time - TimePre;
				if (dfpSol.m_ARRatio > 1.5 || dfpSol.m_AmbState > 2) PassSpan += dt;
				AllSpan += dt;
			}

			TimePre = dfpSol.m_time;
		}

		double rate = PassSpan / (AllSpan + 1.0);
		if (rate < 0.33) m_SINSOpt.m_bTwiceProcLC_NumCheck = false;

 		return true;

	}


	bool CSINSApplication::InsIntegrityCheck()
	{
		GPSTIME  IMUTimePre, IMUTime;
		CSINSOption SINSOpt = m_SINSOpt;
		CIMUFFStream IMUFFStream;
		char msg[MAXSIZE];
		double GapThreshold = 0.5;
		GPSTIME imu_BeginTime = m_IMUDataContainer.m_BeginTime;
		GPSTIME imu_EndTime = m_IMUDataContainer.m_EndTime;
		GPSTIME obs_BeginTime = m_OBSDataContainers[IROVE].m_BeginTime;
		GPSTIME obs_EndTime = m_OBSDataContainers[IROVE].m_EndTime;
		// 1. IMU文件判断
		// 1.1 解码正确判断
		if (imu_BeginTime.GPSWeek < 1 || imu_BeginTime.GPSWeek > 10000 ||
			imu_EndTime.GPSWeek < 1 || imu_EndTime.GPSWeek > 10000)
		{
			string msg("IMU data is abnormal, please check the IMR file!");
			DumpException(msg, LOGT_PRE, DUMP_LOCATION);
			return false;
		}

		// 1.2 IMU数据采集时间和Gnss数据采集时间是否存在重叠区
		if (m_OBSDataContainers[IROVE].m_RecordNum>0&&(obs_BeginTime >= imu_EndTime || obs_EndTime <= imu_BeginTime))
		{
			string msg("There is no overlap between IMU data and GNSS data!");
			DumpException(msg, LOGT_PRE, DUMP_LOCATION);
			return false;
		}

		// 2. 读取SINS数据
		IMUFFStream.m_IMUDataGapTol = SINSOpt.m_IMUDataGapTol;
		IMUFFStream.setDataContainer(&m_IMUDataContainer);
		if (IMUFFStream.init(false) == false) return false;
		IMUFFStream.getOneRecordData();
		while (IMUFFStream.peakOneRecordTime() <= SINSOpt.m_IMUBeginTime) IMUFFStream.getOneRecordData();
		if (IMUFFStream.CheckDataIsEOF()) return false;
		IMUTimePre = IMUFFStream.m_time;
		while (IMUFFStream.getOneRecordData())
		{
			IMUTime = IMUFFStream.m_time;
			//	丢包时间超过0.5s警告
			if (IMUTime - IMUTimePre > GapThreshold)
			{
				//  输出警告/错误信息
				memset(msg, 0, sizeof(char)*MAXSIZE);
				sprintf(msg, "IMU data gap %s(%d,%.4lf)-(%d,%.4lf):%.4lf.", gs_Placeholder.c_str(), 
					IMUTimePre.GPSWeek, _TOW(IMUTimePre), IMUTime.GPSWeek, _TOW(IMUTime), IMUTime - IMUTimePre);

				DumpException(msg, LOGT_PRE, DUMP_LOCATION);
			}
			IMUTimePre = IMUTime;
		}
		return true;
	}


	void CSINSApplication::SystemAutoSelect()
	{
		// 统计流动站/基站的各个系统、各个频点的相位/伪距存在的占比
		double ValidRate_P[2][NSYS][NFREQ] = { 0.0 }, ValidRate_L[2][NSYS][NFREQ] = { 0.0 };

		// 暂存opt信息
		bool bSysUsed[NSYS] = { false };
		for (int i = 0; i < NFREQ; i++)
		{
			if (m_SINSOpt.m_GNSSOpt.m_SatSYS&SYSGPS) bSysUsed[ISYSGPS] = true;
			if (m_SINSOpt.m_GNSSOpt.m_SatSYS&SYSGLO) bSysUsed[ISYSGLO] = true;
			if (m_SINSOpt.m_GNSSOpt.m_SatSYS&SYSBD2) bSysUsed[ISYSBD2] = true;
			if (m_SINSOpt.m_GNSSOpt.m_SatSYS&SYSBD3) bSysUsed[ISYSBD3] = true;
			if (m_SINSOpt.m_GNSSOpt.m_SatSYS&SYSGAL) bSysUsed[ISYSGAL] = true;
			if (m_SINSOpt.m_GNSSOpt.m_SatSYS&SYSQZS) bSysUsed[ISYSQZS] = true;
		}

		// 循环所有观测文件
		for (int i = 0; i < m_OBSDataContainers.size(); i++)
		{
			// 循环不同频点统计观测值占比
			for (int f = 0; f < NFREQ; f++)
			{
				int AllEpoch[NSYS] = { 0 };
				int ValidEpoch_P[NSYS] = { 0 }, ValidEpoch_L[NSYS] = { 0 };

				COBSFFStream OBSFFStream;
				OBSFFStream.setDataContainer(&m_OBSDataContainers[i]);
				if (OBSFFStream.init() == false)
				{
					// 如果有return一定要恢复opt（暂无）
					return;
				}

				// 遍历所有历元
				while (OBSFFStream.getOneRecordData())
				{
					// 循环各个卫星
					for (int j = 0; j < OBSFFStream.m_OBSData.obs.size(); j++)
					{
						int prn = OBSFFStream.m_OBSData.obs[j].prn;

						if (prn <= PRNGLO)
						{
							AllEpoch[ISYSGPS]++;
							if (OBSFFStream.m_OBSData.obs[j].P[f] != 0.0) ValidEpoch_P[ISYSGPS]++;
							if (OBSFFStream.m_OBSData.obs[j].L[f] != 0.0) ValidEpoch_L[ISYSGPS]++;
						}
						else if (PRNGLO < prn && prn <= PRNBD2)
						{
							AllEpoch[ISYSGLO]++;
							if (OBSFFStream.m_OBSData.obs[j].P[f] != 0.0) ValidEpoch_P[ISYSGLO]++;
							if (OBSFFStream.m_OBSData.obs[j].L[f] != 0.0) ValidEpoch_L[ISYSGLO]++;
						}
						else if (PRNBD2 < prn && prn <= PRNBD3)
						{
							AllEpoch[ISYSBD2]++;
							if (OBSFFStream.m_OBSData.obs[j].P[f] != 0.0) ValidEpoch_P[ISYSBD2]++;
							if (OBSFFStream.m_OBSData.obs[j].L[f] != 0.0) ValidEpoch_L[ISYSBD2]++;
						}
						else if (PRNBD3 < prn && prn <= PRNGAL)
						{
							AllEpoch[ISYSBD3]++;
							if (OBSFFStream.m_OBSData.obs[j].P[f] != 0.0) ValidEpoch_P[ISYSBD3]++;
							if (OBSFFStream.m_OBSData.obs[j].L[f] != 0.0) ValidEpoch_L[ISYSBD3]++;
						}
						else if (PRNGAL < prn && prn <= PRNQZS)
						{
							AllEpoch[ISYSGAL]++;
							if (OBSFFStream.m_OBSData.obs[j].P[f] != 0.0) ValidEpoch_P[ISYSGAL]++;
							if (OBSFFStream.m_OBSData.obs[j].L[f] != 0.0) ValidEpoch_L[ISYSGAL]++;
						}
						else if (PRNQZS < prn)
						{
							AllEpoch[ISYSQZS]++;
							if (OBSFFStream.m_OBSData.obs[j].P[f] != 0.0) ValidEpoch_P[ISYSQZS]++;
							if (OBSFFStream.m_OBSData.obs[j].L[f] != 0.0) ValidEpoch_L[ISYSQZS]++;
						}
					}
				}

				for (int j = 0; j < NSYS; j++)
				{
					if (AllEpoch[j] != 0)
					{
						ValidRate_P[i][j][f] = (1.0 * ValidEpoch_P[j]) / (1.0 * AllEpoch[j]);
						ValidRate_L[i][j][f] = (1.0 * ValidEpoch_L[j]) / (1.0 * AllEpoch[j]);
					}
				}
			}
		}

		// 检测所有系统和频点
		int validFreq[NSYS] = { 0 }; // 各系统内有效的频点数量
		int nValidSys = 0;  // 如果是双频及以上，才认为该系统有效
		for (int i = 0; i < NSYS; i++)
		{
			for (int f = 0; f < NFREQ; f++)
			{
				// 流动站和基准站都有2个频点有效，占比阈值为10%
				if (ValidRate_P[0][i][f] > 0.1 && ValidRate_P[1][i][f] > 0.25) validFreq[i]++;
			}
		}

		// 统计有多少频点可用
		for (int i = 0; i < NSYS; i++)
		{
			if (validFreq[i] >= 2 && bSysUsed[i]) nValidSys++;
		}

		// 重新设置解算系统
		if (nValidSys >= 3)  // 至少有3个系统是双频的，才把剩下的系统关闭
		{
			m_SINSOpt.m_GNSSOpt.m_SatSYS = SYSNON;
			if (validFreq[ISYSGPS] >= 2 && bSysUsed[ISYSGPS]) m_SINSOpt.m_GNSSOpt.m_SatSYS |= SYSGPS;
			if (validFreq[ISYSGLO] >= 2 && bSysUsed[ISYSGLO]) m_SINSOpt.m_GNSSOpt.m_SatSYS |= SYSGLO;
			if (validFreq[ISYSBD2] >= 2 && bSysUsed[ISYSBD2]) m_SINSOpt.m_GNSSOpt.m_SatSYS |= SYSBD2;
			if (validFreq[ISYSBD3] >= 2 && bSysUsed[ISYSBD3]) m_SINSOpt.m_GNSSOpt.m_SatSYS |= SYSBD3;
			if (validFreq[ISYSGAL] >= 2 && bSysUsed[ISYSGAL]) m_SINSOpt.m_GNSSOpt.m_SatSYS |= SYSGAL;
			if (validFreq[ISYSQZS] >= 2 && bSysUsed[ISYSQZS]) m_SINSOpt.m_GNSSOpt.m_SatSYS |= SYSQZS;
		}

		return;
	}


	bool CSINSApplication::runSINSApp(string optFile,CSINSOption* pSINSOpt)
	{
		APPINFO AppInfo[2];
		if (m_AppInfo == NULL) m_AppInfo = AppInfo;

		DumpException("IPSDLL: run SINS App -->", LOGT_INFO);
		
		m_OptFile = optFile;

		if(pSINSOpt) m_SINSOpt = *pSINSOpt;
		else if(m_SINSOpt.readOptFile(m_OptFile) == false) return false;

		// 检查所有Opt参数
		if (m_SINSOpt.m_PMode == 2 || m_SINSOpt.m_PMode == 3 || m_SINSOpt.m_PMode == 5)
		{
			if (m_SINSOpt.m_GNSSOpt.CheckAllOption() == false) return false;
		}
		if (m_SINSOpt.getIMUErrorModel() == false) return false;
		if (m_SINSOpt.CheckAllOption() == false) return false;

		if (init() == false) return false;
		if (ReadFiles() == false) return false;

		if (gs_bHCDH || gs_bLJYY)
		{
			GNSSIntegrityCheck();
			InsIntegrityCheck();
			if (m_SINSOpt.m_PMode == 3) SystemAutoSelect();
		}

		if (m_SINSOpt.m_PMode == 2)
		{
			runLC();
		}
		else if (m_SINSOpt.m_PMode == 3 || m_SINSOpt.m_PMode == 5)
		{
			runTC();
		}
		else if (m_SINSOpt.m_PMode == 4)
		{
			runPADLC();
		}
		else if (m_SINSOpt.m_PMode == 1)
		{
			runINS();
		}


		// 保存解算报告
		if (gs_bIPSAPP)
		{
			string filename = m_pGNSSOpt->m_ProjPath + FILEPATHSEP + m_pGNSSOpt->m_ProjName + ".rpt";
			string mode[2] = { "NONE", "NONE" };
			string mode_t = "NONE";
			if (m_SINSOpt.m_PMode == 1)
			{
				mode_t = "INS";
			}
			else if (m_SINSOpt.m_PMode == 2)
			{
				mode_t = "LC";
			}
			else if (m_SINSOpt.m_PMode == 3 || m_SINSOpt.m_PMode == 5)
			{
				if (m_pGNSSOpt->m_PMode == 2) mode_t = "PPP-TC";
				else mode_t = "RTK-TC";
			}
			else if (m_SINSOpt.m_PMode == 4)
			{
				mode_t = "PAD";
			}
			if (m_SINSOpt.m_FilterType != 1) mode[0] = mode_t;
			if (m_SINSOpt.m_FilterType != 0) mode[1] = mode_t;

			char opt_name[MAXSIZE] = { '\0' };
			if (getFileInfo(_C(optFile.c_str()), NULL, NULL, opt_name, NULL))
				UpdateSolRptFile(filename, string(opt_name), mode, m_SINSSolRpt);
		}

		// 最后输出结果对比文件
		ConvertKMLFiles();
		CompareFilesResults();


		gs_DumpExpTime[0].GPSWeek = -1;
		gs_DumpExpTime[1].GPSWeek = -1;
		gs_DumpExpTime[2].GPSWeek = -1;
		DumpException("Please wait for the solution result.", LOGT_INFO);

		//XZ 结果比较
		///< IE-IE
		//int errID = 2;
		//int trueID = 2;
		//int num = 9;
		//int errCols[9] = { 10,11,12,16,17,18,22,23,24 };
		//int trueCols[9] = { 10,11,12,16,17,18,22,23,24 };
	
		//string IE_STIM300 = "H:\\POSMind-Group\\02Data\\Mate_Data\\2023Data\\20230103\\FSAS_Alloy-LC-Forward.imr.ref";
		//string IE_100C = "H:\\POSMind-Group\\02Data\\Mate_Data\\2023Data\\20230103\\ROVE_20230103_01_FSAS-100C.imr.ref";
		//string STIM300_Com = IE_STIM300 + ".com";

		//CompareResultFiles(IE_STIM300, IE_100C, STIM300_Com, errID, trueID, num, errCols, trueCols, 29, 29);

		///<IPS-IE
		/*int errID = 2;
		int trueID = 2;
		int num = 9;
		int errCols[9] = { 10,11,12,16,17,18,22,23,24 };
		int trueCols[9] = { 10,11,12,16,17,18,22,23,24 };

		string IE_STIM300 = "H:\\POSMind-Group\\02Data\\Qiao\\@@2020-05-14-044840\\Qiao_RTK_Forward.ref";
		string IE_100C = "H:\\POSMind-Group\\02Data\\Qiao\\@@2020-05-14-044840\\044827_T.21O.ref";
		string STIM300_Com = IE_STIM300 + ".com";

		CompareResultFiles(IE_STIM300, IE_100C, STIM300_Com, errID, trueID, num, errCols, trueCols, 29, 29);
*/

		return true;
	}


	void CSINSApplication::CompareFilesResults()
	{
#ifdef IPSDLL_Debug_ON
		int errID = 4;
		int trueID = 2;
		int num = 9;
		int errCols[9] = { 5,6,7,27,28,29,39,40,41 };
		int trueCols[9] = { 10,11,12,16,17,18,22,23,24 };

		string flcComFile = m_flfFile + ".com";
		string rlcComFile = m_rlfFile + ".com";
		string flsComFile = m_flsFile + ".com";
		string rlsComFile = m_rlsFile + ".com";
		string dlsComFile = m_dlsFile + ".com";
		string dlfComFile = m_dlfFile + ".com";

		string ftcComFile = m_ftfFile + ".com";
		string rtcComFile = m_rtfFile + ".com";
		string ftsComFile = m_ftsFile + ".com";
		string rtsComFile = m_rtsFile + ".com";
		string dtsComFile = m_dtsFile + ".com";
		string dtfComFile = m_dtfFile + ".com";

		if (m_SINSOpt.m_FilterType == 0 || m_SINSOpt.m_FilterType == 2)
		{	
			if (m_SINSOpt.m_PMode == 3 || m_SINSOpt.m_PMode == 5)
			{
				CompareResultFiles(m_ftfFile, m_TRJFile, ftcComFile, errID, trueID, num, errCols, trueCols, 47, 29);
				CompareResultFiles(m_ftsFile, m_TRJFile, ftsComFile, errID, trueID, num, errCols, trueCols, 47, 29);
			}
			else
			{
				CompareResultFiles(m_flfFile, m_TRJFile, flcComFile, errID, trueID, num, errCols, trueCols, 23, 29);
				CompareResultFiles(m_flsFile, m_TRJFile, flsComFile, errID, trueID, num, errCols, trueCols, 23, 29);
			}
		}

		if (m_SINSOpt.m_FilterType == 1 || m_SINSOpt.m_FilterType == 2)
		{
			if (m_SINSOpt.m_PMode == 3 || m_SINSOpt.m_PMode == 5)
			{
				CompareResultFiles(m_rtfFile, m_TRJFile, rtcComFile, errID, trueID, num, errCols, trueCols, 47, 29);
				CompareResultFiles(m_rtsFile, m_TRJFile, rtsComFile, errID, trueID, num, errCols, trueCols, 47, 29);
			}
			else
			{
				CompareResultFiles(m_rlfFile, m_TRJFile, rlcComFile, errID, trueID, num, errCols, trueCols, 23, 29);
				CompareResultFiles(m_rlsFile, m_TRJFile, rlsComFile, errID, trueID, num, errCols, trueCols, 23, 29);
			}	
		}

		if (m_SINSOpt.m_FilterType == 2)
		{
			if (m_SINSOpt.m_PMode == 3 || m_SINSOpt.m_PMode == 5)
			{
				CompareResultFiles(m_dtsFile, m_TRJFile, dtsComFile, errID, trueID, num, errCols, trueCols, 47, 29);
				CompareResultFiles(m_dtfFile, m_TRJFile, dtfComFile, errID, trueID, num, errCols, trueCols, 47, 29);
			}
			else
			{
				CompareResultFiles(m_dlsFile, m_TRJFile, dlsComFile, errID, trueID, num, errCols, trueCols, 23, 29);
				CompareResultFiles(m_dlfFile, m_TRJFile, dlfComFile, errID, trueID, num, errCols, trueCols, 23, 29);
			}	
		}

		if (gs_bHCDH)
		{
			num = 3;
			errID = 28;
			trueID = 2;
			int errCols_Pos[3] = { 2,3,4 };
			int trueCols_Pos[3] = { 10,11,12 };
			int errCols_Att[3] = { 7,6,5 };
			int trueCols_Att[3] = { 22,23,24 };

			char cFileTitle[1024] = "";
			getFileInfo(_C(m_IMUFile.c_str()), cFileTitle, NULL, NULL, NULL);
			string ExportFile = m_FilePath + FILEPATHSEP + cFileTitle + ".PosT";
			string PosComFile = ExportFile + ".PosCom";
			string AttComFile = ExportFile + ".AttCom";

			string TRJFile = m_TRJFile + "_HighFreq";

			CompareResultFiles_Pos(ExportFile, TRJFile, PosComFile, errID, trueID, num, errCols_Pos, trueCols_Pos, 3, 29);
			CompareResultFiles_Att(ExportFile, TRJFile, AttComFile, errID, trueID, num, errCols_Att, trueCols_Att, 3, 29);
		}
#endif
	}


	void CSINSApplication::ConvertKMLFiles()
	{
		if (m_SINSOpt.m_FilterType == 0 || m_SINSOpt.m_FilterType == 2)
		{
			if (m_SINSOpt.m_PMode == 3 || m_SINSOpt.m_PMode == 5)
			{
				if (AccessFile(m_ftfFile)) ConvertKML(2, m_ftfFile, m_ftfFile + ".kml");
				if (AccessFile(m_ftsFile)) ConvertKML(2, m_ftsFile, m_ftsFile + ".kml");
			}
			else
			{
				if (AccessFile(m_flfFile)) ConvertKML(2, m_flfFile, m_flfFile + ".kml");
				if (AccessFile(m_flsFile)) ConvertKML(2, m_flsFile, m_flsFile + ".kml");
			}
		}

		if (m_SINSOpt.m_FilterType == 1 || m_SINSOpt.m_FilterType == 2)
		{
			if (m_SINSOpt.m_PMode == 3 || m_SINSOpt.m_PMode == 5)
			{
				if (AccessFile(m_rtfFile)) ConvertKML(2, m_rtfFile, m_rtfFile + ".kml");
				if (AccessFile(m_rtsFile)) ConvertKML(2, m_rtsFile, m_rtsFile + ".kml");
			}
			else
			{
				if (AccessFile(m_rlfFile)) ConvertKML(2, m_rlfFile, m_rlfFile + ".kml");
				if (AccessFile(m_rlsFile)) ConvertKML(2, m_rlsFile, m_rlsFile + ".kml");
			}
		}

		if (m_SINSOpt.m_FilterType == 2)
		{
			if (m_SINSOpt.m_PMode == 3 || m_SINSOpt.m_PMode == 5)
			{
				if (AccessFile(m_dtsFile)) ConvertKML(2, m_dtsFile, m_dtsFile + ".kml");
				if (AccessFile(m_dtfFile)) ConvertKML(2, m_dtfFile, m_dtfFile + ".kml");
			}
			else
			{
				if (AccessFile(m_dlsFile)) ConvertKML(2, m_dlsFile, m_dlsFile + ".kml");
				if (AccessFile(m_dlfFile)) ConvertKML(2, m_dlfFile, m_dlfFile + ".kml");
			}
		}
	}
}
