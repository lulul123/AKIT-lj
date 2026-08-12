/**
* @file
* @brief    Struct, Define and Const (SDC)
* @details  This file contains Struct, Define and Const (SDC) \n
* @author   Feng Zhu. Email: fzhu@whu.edu.cn
* @date     2018/01/17
* @version  2.3.20230417
* @par      Copyright(c) 2012-2021 School of Geodesy and Geomatics, University of Wuhan. All Rights Reserved.
*           POSMind Software Kernel, named IPS (Inertial Plus Sensing), this file is a part of IPS.
* @par      History:
*           2018/01/17,Feng Zhu, new \n
*/


#ifndef BASETK_BASESDC_H
#define BASETK_BASESDC_H

#include <math.h>
#include <time.h>
#include <memory.h>
#include <string>
#include <vector>
#include <list>
#include <map>
#include <algorithm>
#include <functional>

using namespace std;


namespace basetk
{

/**********************************************************
* dubug版本和release版本对变量的初始化不同,变量的截断误差也不同
* 所有变量一定要赋初值,STL模板在其它系统下可能有问题
* long在不同系统下的字节数不一样,切记不要用sizeof(long),应用sizeof(int)
***********************************************************/


/*============================ start define ======================================*/

//#define _WINDOWS			///< Windows下打开
//#define _LINUX			///< Linux/Android/iOS下打开
//#define _ANDROID			///< Android下打开
//#define _APPLE_iOS		///< iOS下打开

#ifdef _WINDOWS
#define FILEPATHSEP			"\\"
#define cFILEPATHSEP	    '\\'
#else
#define FILEPATHSEP			"/"
#define cFILEPATHSEP	    '/'
#endif

#define IPSDLL_EXPORT_ON                                            ///< 是否导出DLL
#define IPSDLL_REALTIME_OFF                                         ///< 是否为实时系统
#define IPSDLL_Debug_ON                                             ///< 是否为调试模式（非调试模式将关闭所有文本调试信息）
const static bool gs_bIPSAPP = false;                               ///< IPS界面输出
const static bool gs_bUsedPThread = false;                          ///< 是否使用多线程,只有WINDOS下使用多线程

///< 客户定制化开关
const static bool gs_bNFGT = false;                                 ///< 是否为南方高铁解算
const static bool gs_bLJTY = false;                                 ///< 是否为砺剑天眼解算
const static bool gs_bHCDH = false;                                  ///< 是否为华测导航解算
const static bool gs_bLJYY = false;                                 ///< 是否为珞珈依云解算

#define IPSNAME             "POSMind Software (IPS2.3)"				///< 软件名字
#define IPSDLL_DEBUG                                                ///< 所有调试的代码都放入预定义块
#define IPS_EPSILON         2.2204460492503131e-016                 ///< epsilon == DBL_EPSILON
#define MAXSIZE             (2048)
#define MAXDIST             (100)                                   ///< max freedom of statistics distribution
#define ZeroTolerate        (1e-15)                                 ///< tolerate of zero
#define PI                  (3.1415926535897932384626433832795)     ///< pi
#define PI_2                (6.2831853071795864769252867665590)     ///< 2*pi
#define PI_12               (1.5707963267948966192313216916398)     ///< 0.5*pi
#define D2R                 (0.0174532925199432957692369076849)     ///< deg to rad
#define R2D                 (57.295779513082322864647721871734)     ///< rad to deg
#define CLIGHT              (299792458.0)                           ///< speed of light (m/s)
#define AS2R                (D2R/3600.0)                            ///< arc sec to radian 
#define COMMENTH            '#'                                     ///< comment line indicator for solution
#define SQR(x)              ((x)*(x))                               ///< x*x
#define MAXCHISQ            (100)                                   ///< max number of chi-sqr
#define ROUND(x)            (int)floor((x)+0.5)                     ///< round x
#define ZeroStruct(x,type)  (memset(&x,0,sizeof(type)))             ///< clear struct
#define isEqual(a,b)        (fabs(a-b)<IPS_EPSILON ? true : false)  ///< a == b
#define _TOW(gt)            ((gt).secsOfWeek + (gt).fracOfSec)      ///< time of week
#define _SEC(gt)            ((gt).GPSWeek*604800.0 + _TOW(gt))      ///< gt -> sec
#define _FRAC(x)            ((x)-int(x))                            ///< Fraction of x
#define _IL(mI,mL)          {I=(mI)+(mL);}                          ///< index + length
#define _P(vec)             (&(vec[0]))                             ///< point of vec[0]
#define _C(x)               (const_cast<char *>(x))                 ///< const (x) to non-const (x)
#define _I(itr)             (&(*(itr)))                             ///< point of iteration
#define MIN(x,y)            ((x)<=(y)?(x):(y))
#define MAX(x,y)            ((x)>=(y)?(x):(y))
#define ALFA                (1.0e-8)                                ///< IGG3抗差中,粗差的权
#define FTP_TIMEOUT         60                                      ///< ftp timeout (s)
#define FTP_RETRY           3                                       ///< ftp number of retry
#define SWAP(x,y)           {double z=x; x=y; y=z;}
#define _BaseFileName(Path,Name,Type) ((Path) + ((Path)[(Path).length()-1] == cFILEPATHSEP ? "" : FILEPATHSEP) + (Name) + "." + (Type))
#define FILE_IMUMODEL       ("IPSSYSTEM/IMU/IMUErrorModel.txt")
#define FILE_GNSSLEVERARM   ("IPSSYSTEM/IMU/GNSS_LeverArm.txt") 
#define FILE_ExportDefault  ("IPSSYSTEM/EXP/ExportDefault.xml")  
#define FILE_BASEXYZ        ("IPSSYSTEM/ATX/BaseXYZ.txt")
#define FILE_ANTENNAINF     ("IPSSYSTEM/ATX/GNSS_AntennaInf.txt")
#define FILE_GNSSCODEPRIOR  ("IPSSYSTEM/GNSS_CodePrior.txt")   

///< ToDo(IPS2.3):
// 注意，如果有NHC或估计了安装误差，这里千万不能开OFF
#define CORRECT_ROTANGLE_IMU_ON                                     ///< 是否在IMU原始数据上进行安置角改正 ON:是 OFF:否 

#if defined (IPSDLL_EXPORT_ON) && defined (_WINDOWS)
#define IPS_DLLEXPORT __declspec(dllexport)                         ///< 定义DLL导出
#pragma warning(disable:4251)                                       ///< 把警告屏蔽
#else
#define IPS_DLLEXPORT
#endif																					


///< 导航定位的载体类型
enum IPS_USER_TYPE {
	IPSUSER_NONE,				///< 默认
	IPSUSER_CAR,				///< 车载
	IPSUSER_AIR,				///< 机载
	IPSUSER_SHIP,				///< 船载
	IPSUSER_UGV,				///< 地面低速无人车/机器人
	IPSUSER_UAV,				///< 低空/小型无人机
	IPSUSER_USV,				///< 小型无人船
	IPSUSER_Train,				///< 火车/高铁
	IPSUSER_Pedestrian,			///< 行人
};

///< 导航定位的观测场景
enum IPS_SCENE_TYPE {
	IPSSCENE_NONE,				///< 默认
	IPSSCENE_G_OpenSky,			///< 地面(G):开阔环境
	IPSSCENE_G_UrbanCanyon,		///< 地面(G):城市峡谷
	IPSSCENE_G_Tree,			///< 地面(G):林荫/树木
	IPSSCENE_G_Overpass,		///< 地面(G):城市高架
	IPSSCENE_G_Tunnel,			///< 地面(G):隧道

	IPSSCENE_A_OpenSky,			///< 空中(A):开阔环境
	IPSSCENE_A_UrbanCanyon,		///< 空中(A):城市峡谷

	IPSSCENE_S_OpenSky,			///< 海洋(S):水上环境
	IPSSCENE_S_Underwater,		///< 海洋(S):水下环境
};

/*============================ end define ========================================*/






/*============================ start const =======================================*/
///< global static = gs_ : 全局静态变量

static const double gs_WGS84_a = 6378137.0;                     ///< earth semimajor axis (WGS84) (m)
static const double gs_WGS84_b = 6356752.31425;                 ///< earth semimajor axis (WGS84) (m)
static const double gs_WGS84_FE = 1.0 / 298.257223563;             ///< earth flattening (WGS84)
static const double gs_WGS84_e2 = 2 * gs_WGS84_FE - SQR(gs_WGS84_FE);
static const double gs_WGS84_e = sqrt(gs_WGS84_e2);             ///< earth first eccentricity
static const double gs_WGS84_GME = 3.986004418E+14;               ///< earth gravitational constant (m^3/s^2) fM
static const double gs_WGS84_J2 = 1.08263E-03;                   ///<
static const double gs_WGS84_J4 = -2.37091222E-6;                 ///<
static const double gs_WGS84_J6 = 6.08347E-9;                    ///<
static const double gs_WGS84_OMGE = 7.2921151467E-5;               ///< earth angular velocity (IS-GPS) (rad/s)
static const double gs_WGS84_Ge = 9.7803267714;                  ///< gravity at equator (m/s^2) 
static const double gs_WGS84_Gp = 9.8322011865;                  ///< gravity at polar   (m/s^2) 
static const double gs_WGS84_Gm = 9.7976446561;                  ///< Mean value (normal) gravity (m/s^2)
static const double gs_WGS84_Galph = 0.00193336138707;              ///< gravity formula constant

static const double gs_CGCS2000_a = 6378137.0;                      ///< earth semimajor axis (WGS84) (m)
static const double gs_CGCS2000_b = 6356752.314140356;              ///< earth semimajor axis (WGS84) (m)
static const double gs_CGCS2000_FE = 1.0 / 298.257222101;           ///< earth flattening (WGS84)
static const double gs_CGCS2000_e2 = 2 * gs_CGCS2000_FE - SQR(gs_CGCS2000_FE);
static const double gs_CGCS2000_e = sqrt(gs_CGCS2000_e2);           ///< earth first eccentricity
static const double gs_CGCS2000_GME = 3.986004418E+14;              ///< earth gravitational constant (m^3/s^2) fM
static const double gs_CGCS2000_OMGE = 7.2921150E-5;                ///< earth angular velocity (IS-GPS) (rad/s)

static const double gs_min = D2R / 60.0;                              ///< 一个单位角分 对应的 弧度
static const double gs_sec = D2R / 3600.0;                            ///< 一个单位角秒 对应的 弧度
static const double gs_dps = D2R;                                   ///< 1度/秒 对应的 弧度/秒
static const double gs_dph = D2R / 3600.0;                            ///< 1度/小时 对应的 弧度/秒
static const double gs_dpss = D2R;                                   ///< 1度/xsqrt(秒) 对应的 弧度/xsqrt(秒)
static const double gs_dpsh = D2R / 60.0;                              ///< 1度/xsqrt(小时) 对应的 弧度/xsqrt(秒), 角度随机游走系数
static const double gs_g0 = gs_WGS84_Gm;                           ///< 重力加速度(米/秒平方)
static const double gs_ug = gs_g0 * 1e-6;                          ///< 毫重力加速度
static const double gs_mg = gs_g0 * 1e-3;                          ///< 微重力加速度
static const double gs_ppm = 1e-6;                                  ///< part per million(百万分之一)

#ifdef _WINDOWS
static const double gs_epsilon = std::numeric_limits<double>::epsilon() * 100;
#else
static const double gs_epsilon = IPS_EPSILON * 100;
#endif

static const int    gs_MaxOutFile = 30;

///< 卡方分布检验gs_chisqr_0pxxx, xxx表示xx.x%的显著性水平,即xx.x%的小概率事件发生了

const double gs_chisqr_0p999[100] = {
	0.00, 0.00, 0.02, 0.09, 0.21, 0.38, 0.60, 0.86, 1.15, 1.48,
	1.83, 2.21, 2.62, 3.04, 3.48, 3.94, 4.42, 4.90, 5.41, 5.92,
	6.45, 6.98, 7.53, 8.08, 8.65, 9.22, 9.80, 10.39, 10.99, 11.59,
	12.20, 12.81, 13.43, 14.06, 14.69, 15.32, 15.97, 16.61, 17.26, 17.92,
	18.58, 19.24, 19.91, 20.58, 21.25, 21.93, 22.61, 23.29, 23.98, 24.67,
	25.37, 26.07, 26.76, 27.47, 28.17, 28.88, 29.59, 30.30, 31.02, 31.74,
	32.46, 33.18, 33.91, 34.63, 35.36, 36.09, 36.83, 37.56, 38.30, 39.04,
	39.78, 40.52, 41.26, 42.01, 42.76, 43.51, 44.26, 45.01, 45.76, 46.52,
	47.28, 48.04, 48.80, 49.56, 50.32, 51.08, 51.85, 52.62, 53.39, 54.16,
	54.93, 55.70, 56.47, 57.25, 58.02, 58.80, 59.58, 60.36, 61.14, 61.92
};

const double gs_chisqr_0p990[100] = {
	0.00, 0.02, 0.11, 0.30, 0.55, 0.87, 1.24, 1.65, 2.09, 2.56,
	3.05, 3.57, 4.11, 4.66, 5.23, 5.81, 6.41, 7.01, 7.63, 8.26,
	8.90, 9.54, 10.20, 10.86, 11.52, 12.20, 12.88, 13.56, 14.26, 14.95,
	15.66, 16.36, 17.07, 17.79, 18.51, 19.23, 19.96, 20.69, 21.43, 22.16,
	22.91, 23.65, 24.40, 25.15, 25.90, 26.66, 27.42, 28.18, 28.94, 29.71,
	30.48, 31.25, 32.02, 32.79, 33.57, 34.35, 35.13, 35.91, 36.70, 37.48,
	38.27, 39.06, 39.86, 40.65, 41.44, 42.24, 43.04, 43.84, 44.64, 45.44,
	46.25, 47.05, 47.86, 48.67, 49.48, 50.29, 51.10, 51.91, 52.72, 53.54,
	54.36, 55.17, 55.99, 56.81, 57.63, 58.46, 59.28, 60.10, 60.93, 61.75,
	62.58, 63.41, 64.24, 65.07, 65.90, 66.73, 67.56, 68.40, 69.23, 70.06,
};

const double gs_chisqr_0p950[100] = {
	0.00, 0.10, 0.35, 0.71, 1.15, 1.64, 2.17, 2.73, 3.33, 3.94,
	4.57, 5.23, 5.89, 6.57, 7.26, 7.96, 8.67, 9.39, 10.12, 10.85,
	11.59, 12.34, 13.09, 13.85, 14.61, 15.38, 16.15, 16.93, 17.71, 18.49,
	19.28, 20.07, 20.87, 21.66, 22.47, 23.27, 24.07, 24.88, 25.70, 26.51,
	27.33, 28.14, 28.96, 29.79, 30.61, 31.44, 32.27, 33.10, 33.93, 34.76,
	35.60, 36.44, 37.28, 38.12, 38.96, 39.80, 40.65, 41.49, 42.34, 43.19,
	44.04, 44.89, 45.74, 46.59, 47.45, 48.31, 49.16, 50.02, 50.88, 51.74,
	52.60, 53.46, 54.33, 55.19, 56.05, 56.92, 57.79, 58.65, 59.52, 60.39,
	61.26, 62.13, 63.00, 63.88, 64.75, 65.62, 66.50, 67.37, 68.25, 69.13,
	70.00, 70.88, 71.76, 72.64, 73.52, 74.40, 75.28, 76.16, 77.05, 77.93,
};

const double gs_chisqr_0p900[100] = {
	0.02, 0.21, 0.58, 1.06, 1.61, 2.20, 2.83, 3.49, 4.17, 4.87,
	5.58, 6.30, 7.04, 7.79, 8.55, 9.31, 10.09, 10.86, 11.65, 12.44,
	13.24, 14.04, 14.85, 15.66, 16.47, 17.29, 18.11, 18.94, 19.77, 20.60,
	21.43, 22.27, 23.11, 23.95, 24.80, 25.64, 26.49, 27.34, 28.20, 29.05,
	29.91, 30.77, 31.63, 32.49, 33.35, 34.22, 35.08, 35.95, 36.82, 37.69,
	38.56, 39.43, 40.31, 41.18, 42.06, 42.94, 43.82, 44.70, 45.58, 46.46,
	47.34, 48.23, 49.11, 50.00, 50.88, 51.77, 52.66, 53.55, 54.44, 55.33,
	56.22, 57.11, 58.01, 58.90, 59.79, 60.69, 61.59, 62.48, 63.38, 64.28,
	65.18, 66.08, 66.98, 67.88, 68.78, 69.68, 70.58, 71.48, 72.39, 73.29,
	74.20, 75.10, 76.01, 76.91, 77.82, 78.73, 79.63, 80.54, 81.45, 82.36
};

const double gs_chisqr_0p800[100] = {
	0.06, 0.45, 1.01, 1.65, 2.34, 3.07, 3.82, 4.59, 5.38, 6.18,
	6.99, 7.81, 8.63, 9.47, 10.31, 11.15, 12.00, 12.86, 13.72, 14.58,
	15.44, 16.31, 17.19, 18.06, 18.94, 19.82, 20.70, 21.59, 22.48, 23.36,
	24.26, 25.15, 26.04, 26.94, 27.84, 28.73, 29.64, 30.54, 31.44, 32.34,
	33.25, 34.16, 35.07, 35.97, 36.88, 37.80, 38.71, 39.62, 40.53, 41.45,
	42.36, 43.28, 44.20, 45.12, 46.04, 46.96, 47.88, 48.80, 49.72, 50.64,
	51.56, 52.49, 53.41, 54.34, 55.26, 56.19, 57.11, 58.04, 58.97, 59.90,
	60.83, 61.76, 62.69, 63.62, 64.55, 65.48, 66.41, 67.34, 68.27, 69.21,
	70.14, 71.07, 72.01, 72.94, 73.88, 74.81, 75.75, 76.69, 77.62, 78.56,
	79.50, 80.43, 81.37, 82.31, 83.25, 84.19, 85.13, 86.07, 87.01, 87.95
};

const double gs_chisqr_0p750[100] = {
	0.10, 0.58, 1.21, 1.92, 2.67, 3.45, 4.25, 5.07, 5.90, 6.74,
	7.58, 8.44, 9.30, 10.17, 11.04, 11.91, 12.79, 13.68, 14.56, 15.45,
	16.34, 17.24, 18.14, 19.04, 19.94, 20.84, 21.75, 22.66, 23.57, 24.48,
	25.39, 26.30, 27.22, 28.14, 29.05, 29.97, 30.89, 31.81, 32.74, 33.66,
	34.58, 35.51, 36.44, 37.36, 38.29, 39.22, 40.15, 41.08, 42.01, 42.94,
	43.87, 44.81, 45.74, 46.68, 47.61, 48.55, 49.48, 50.42, 51.36, 52.29,
	53.23, 54.17, 55.11, 56.05, 56.99, 57.93, 58.87, 59.81, 60.76, 61.70,
	62.64, 63.58, 64.53, 65.47, 66.42, 67.36, 68.31, 69.25, 70.20, 71.14,
	72.09, 73.04, 73.99, 74.93, 75.88, 76.83, 77.78, 78.73, 79.68, 80.62,
	81.57, 82.52, 83.47, 84.42, 85.38, 86.33, 87.28, 88.23, 89.18, 90.13,
};

const double gs_chisqr_0p600[100] = {
	0.27, 1.02, 1.87, 2.75, 3.66, 4.57, 5.49, 6.42, 7.36, 8.30,
	9.24, 10.18, 11.13, 12.08, 13.03, 13.98, 14.94, 15.89, 16.85, 17.81,
	18.77, 19.73, 20.69, 21.65, 22.62, 23.58, 24.54, 25.51, 26.48, 27.44,
	28.41, 29.38, 30.34, 31.31, 32.28, 33.25, 34.22, 35.19, 36.16, 37.13,
	38.11, 39.08, 40.05, 41.02, 42.00, 42.97, 43.94, 44.92, 45.89, 46.86,
	47.84, 48.81, 49.79, 50.76, 51.74, 52.71, 53.69, 54.67, 55.64, 56.62,
	57.60, 58.57, 59.55, 60.53, 61.51, 62.48, 63.46, 64.44, 65.42, 66.40,
	67.37, 68.35, 69.33, 70.31, 71.29, 72.27, 73.25, 74.23, 75.21, 76.19,
	77.17, 78.15, 79.13, 80.11, 81.09, 82.07, 83.05, 84.03, 85.01, 85.99,
	86.97, 87.95, 88.94, 89.92, 90.90, 91.88, 92.86, 93.84, 94.83, 95.81
};

const double gs_chisqr_0p500[100] = {
	0.45, 1.39, 2.37, 3.36, 4.35, 5.35, 6.35, 7.34, 8.34, 9.34,
	10.34, 11.34, 12.34, 13.34, 14.34, 15.34, 16.34, 17.34, 18.34, 19.34,
	20.34, 21.34, 22.34, 23.34, 24.34, 25.34, 26.34, 27.34, 28.34, 29.34,
	30.34, 31.34, 32.34, 33.34, 34.34, 35.34, 36.34, 37.34, 38.34, 39.34,
	40.34, 41.34, 42.34, 43.34, 44.34, 45.34, 46.34, 47.34, 48.33, 49.33,
	50.33, 51.33, 52.33, 53.33, 54.33, 55.33, 56.33, 57.33, 58.33, 59.33,
	60.33, 61.33, 62.33, 63.33, 64.33, 65.33, 66.33, 67.33, 68.33, 69.33,
	70.33, 71.33, 72.33, 73.33, 74.33, 75.33, 76.33, 77.33, 78.33, 79.33,
	80.33, 81.33, 82.33, 83.33, 84.33, 85.33, 86.33, 87.33, 88.33, 89.33,
	90.33, 91.33, 92.33, 93.33, 94.33, 95.33, 96.33, 97.33, 98.33, 99.33
};

const double gs_chisqr_0p330[100] = {
	0.95, 2.22, 3.43, 4.61, 5.76, 6.90, 8.03, 9.15, 10.26, 11.36,
	12.46, 13.56, 14.65, 15.73, 16.82, 17.90, 18.97, 20.05, 21.12, 22.20,
	23.27, 24.33, 25.40, 26.47, 27.53, 28.59, 29.65, 30.71, 31.77, 32.83,
	33.89, 34.94, 36.00, 37.05, 38.11, 39.16, 40.21, 41.26, 42.31, 43.36,
	44.41, 45.46, 46.51, 47.56, 48.60, 49.65, 50.69, 51.74, 52.79, 53.83,
	54.87, 55.92, 56.96, 58.00, 59.05, 60.09, 61.13, 62.17, 63.21, 64.25,
	65.29, 66.33, 67.37, 68.41, 69.45, 70.49, 71.53, 72.57, 73.60, 74.64,
	75.68, 76.71, 77.75, 78.79, 79.82, 80.86, 81.90, 82.93, 83.97, 85.00,
	86.04, 87.07, 88.11, 89.14, 90.17, 91.21, 92.24, 93.27, 94.31, 95.34,
	96.37, 97.41, 98.44, 99.47, 100.50, 101.53, 102.57, 103.60, 104.63, 105.66
};

const double gs_chisqr_0p250[100] =
{
	1.32, 2.77, 4.11, 5.39, 6.63, 7.84, 9.04, 10.22, 11.39, 12.55,
	13.70, 14.85, 15.98, 17.12, 18.25, 19.37, 20.49, 21.60, 22.72, 23.83,
	24.93, 26.04, 27.14, 28.24, 29.34, 30.43, 31.53, 32.62, 33.71, 34.80,
	35.89, 36.97, 38.06, 39.14, 40.22, 41.30, 42.38, 43.46, 44.54, 45.62,
	46.69, 47.77, 48.84, 49.91, 50.98, 52.06, 53.13, 54.20, 55.27, 56.33,
	57.40, 58.47, 59.53, 60.60, 61.66, 62.73, 63.79, 64.86, 65.92, 66.98,
	68.04, 69.10, 70.16, 71.23, 72.28, 73.34, 74.40, 75.46, 76.52, 77.58,
	78.63, 79.69, 80.75, 81.80, 82.86, 83.91, 84.97, 86.02, 87.08, 88.13,
	89.18, 90.24, 91.29, 92.34, 93.39, 94.45, 95.50, 96.55, 97.60, 98.65,
	99.70, 100.75, 101.80, 102.85, 103.90, 104.95, 106.00, 107.05, 108.09, 109.14
};

const double gs_chisqr_0p200[100] =
{
	1.64, 3.22, 4.64, 5.99, 7.29, 8.56, 9.80, 11.03, 12.24, 13.44,
	14.63, 15.81, 16.98, 18.15, 19.31, 20.47, 21.61, 22.76, 23.90, 25.04,
	26.17, 27.30, 28.43, 29.55, 30.68, 31.79, 32.91, 34.03, 35.14, 36.25,
	37.36, 38.47, 39.57, 40.68, 41.78, 42.88, 43.98, 45.08, 46.17, 47.27,
	48.36, 49.46, 50.55, 51.64, 52.73, 53.82, 54.91, 55.99, 57.08, 58.16,
	59.25, 60.33, 61.41, 62.50, 63.58, 64.66, 65.74, 66.82, 67.89, 68.97,
	70.05, 71.13, 72.20, 73.28, 74.35, 75.42, 76.50, 77.57, 78.64, 79.71,
	80.79, 81.86, 82.93, 84.00, 85.07, 86.13, 87.20, 88.27, 89.34, 90.41,
	91.47, 92.54, 93.60, 94.67, 95.73, 96.80, 97.86, 98.93, 99.99, 101.05,
	102.12, 103.18, 104.24, 105.30, 106.36, 107.43, 108.49, 109.55, 110.61, 111.67
};

const double gs_chisqr_0p100[100] = {
	2.71, 4.61, 6.25, 7.78, 9.24, 10.64, 12.02, 13.36, 14.68, 15.99,
	17.28, 18.55, 19.81, 21.06, 22.31, 23.54, 24.77, 25.99, 27.20, 28.41,
	29.62, 30.81, 32.01, 33.20, 34.38, 35.56, 36.74, 37.92, 39.09, 40.26,
	41.42, 42.58, 43.75, 44.90, 46.06, 47.21, 48.36, 49.51, 50.66, 51.81,
	52.95, 54.09, 55.23, 56.37, 57.51, 58.64, 59.77, 60.91, 62.04, 63.17,
	64.30, 65.42, 66.55, 67.67, 68.80, 69.92, 71.04, 72.16, 73.28, 74.40,
	75.51, 76.63, 77.75, 78.86, 79.97, 81.09, 82.20, 83.31, 84.42, 85.53,
	86.64, 87.74, 88.85, 89.96, 91.06, 92.17, 93.27, 94.37, 95.48, 96.58,
	97.68, 98.78, 99.88, 100.98, 102.08, 103.18, 104.28, 105.37, 106.47, 107.57,
	108.66, 109.76, 110.85, 111.94, 113.04, 114.13, 115.22, 116.32, 117.41, 118.50
};

const double gs_chisqr_0p050[100] = {
	3.84, 5.99, 7.81, 9.49, 11.07, 12.59, 14.07, 15.51, 16.92, 18.31,
	19.68, 21.03, 22.36, 23.68, 25.00, 26.30, 27.59, 28.87, 30.14, 31.41,
	32.67, 33.92, 35.17, 36.42, 37.65, 38.89, 40.11, 41.34, 42.56, 43.77,
	44.99, 46.19, 47.40, 48.60, 49.80, 51.00, 52.19, 53.38, 54.57, 55.76,
	56.94, 58.12, 59.30, 60.48, 61.66, 62.83, 64.00, 65.17, 66.34, 67.50,
	68.67, 69.83, 70.99, 72.15, 73.31, 74.47, 75.62, 76.78, 77.93, 79.08,
	80.23, 81.38, 82.53, 83.68, 84.82, 85.96, 87.11, 88.25, 89.39, 90.53,
	91.67, 92.81, 93.95, 95.08, 96.22, 97.35, 98.48, 99.62, 100.75, 101.88,
	103.01, 104.14, 105.27, 106.39, 107.52, 108.65, 109.77, 110.90, 112.02, 113.15,
	114.27, 115.39, 116.51, 117.63, 118.75, 119.87, 120.99, 122.11, 123.23, 124.34
};

const double gs_chisqr_0p010[100] =
{
	6.63, 9.21, 11.34, 13.28, 15.09, 16.81, 18.48, 20.09, 21.67, 23.21,
	24.72, 26.22, 27.69, 29.14, 30.58, 32.00, 33.41, 34.81, 36.19, 37.57,
	38.93, 40.29, 41.64, 42.98, 44.31, 45.64, 46.96, 48.28, 49.59, 50.89,
	52.19, 53.49, 54.78, 56.06, 57.34, 58.62, 59.89, 61.16, 62.43, 63.69,
	64.95, 66.21, 67.46, 68.71, 69.96, 71.20, 72.44, 73.68, 74.92, 76.15,
	77.39, 78.62, 79.84, 81.07, 82.29, 83.51, 84.73, 85.95, 87.17, 88.38,
	89.59, 90.80, 92.01, 93.22, 94.42, 95.63, 96.83, 98.03, 99.23, 100.43,
	101.62, 102.82, 104.01, 105.20, 106.39, 107.58, 108.77, 109.96, 111.14, 112.33,
	113.51, 114.69, 115.88, 117.06, 118.24, 119.41, 120.59, 121.77, 122.94, 124.12,
	125.29, 126.46, 127.63, 128.80, 129.97, 131.14, 132.31, 133.48, 134.64, 135.81
};

const double gs_chisqr_0p005[100] =
{
	7.88, 10.60, 12.84, 14.86, 16.75, 18.55, 20.28, 21.95, 23.59, 25.19,
	26.76, 28.30, 29.82, 31.32, 32.80, 34.27, 35.72, 37.16, 38.58, 40.00,
	41.40, 42.80, 44.18, 45.56, 46.93, 48.29, 49.64, 50.99, 52.34, 53.67,
	55.00, 56.33, 57.65, 58.96, 60.27, 61.58, 62.88, 64.18, 65.48, 66.77,
	68.05, 69.34, 70.62, 71.89, 73.17, 74.44, 75.70, 76.97, 78.23, 79.49,
	80.75, 82.00, 83.25, 84.50, 85.75, 86.99, 88.24, 89.48, 90.72, 91.95,
	93.19, 94.42, 95.65, 96.88, 98.11, 99.33, 100.55, 101.78, 103.00, 104.21,
	105.43, 106.65, 107.86, 109.07, 110.29, 111.50, 112.70, 113.91, 115.12, 116.32,
	117.52, 118.73, 119.93, 121.13, 122.32, 123.52, 124.72, 125.91, 127.11, 128.30,
	129.49, 130.68, 131.87, 133.06, 134.25, 135.43, 136.62, 137.80, 138.99, 140.17
};

const double gs_chisqr_0p001[100] =
{
	10.83, 13.82, 16.27, 18.47, 20.52, 22.46, 24.32, 26.12, 27.88, 29.59,
	31.26, 32.91, 34.53, 36.12, 37.70, 39.25, 40.79, 42.31, 43.82, 45.31,
	46.80, 48.27, 49.73, 51.18, 52.62, 54.05, 55.48, 56.89, 58.30, 59.70,
	61.10, 62.49, 63.87, 65.25, 66.62, 67.99, 69.35, 70.70, 72.05, 73.40,
	74.74, 76.08, 77.42, 78.75, 80.08, 81.40, 82.72, 84.04, 85.35, 86.66,
	87.97, 89.27, 90.57, 91.87, 93.17, 94.46, 95.75, 97.04, 98.32, 99.61,
	100.89, 102.17, 103.44, 104.72, 105.99, 107.26, 108.53, 109.79, 111.06, 112.32,
	113.58, 114.84, 116.09, 117.35, 118.60, 119.85, 121.10, 122.35, 123.59, 124.84,
	126.08, 127.32, 128.56, 129.80, 131.04, 132.28, 133.51, 134.75, 135.98, 137.21,
	138.44, 139.67, 140.89, 142.12, 143.34, 144.57, 145.79, 147.01, 148.23, 149.45
};

/*============================ end const =========================================*/






/*============================ start struct ======================================*/

typedef struct IPS_DLLEXPORT tagGPSTIME   ///< GPS时,不能改动,否则和IPSDLL中定义不一致,导致错误
{
	int    GPSWeek;
	int    secsOfWeek;
	double fracOfSec;

	tagGPSTIME()
	{
		GPSWeek = -1;
		secsOfWeek = 0;
		fracOfSec = 0.0;
	}

	tagGPSTIME(int week, double sec)
	{
		GPSWeek = week;
		secsOfWeek = int(sec);
		fracOfSec = sec - secsOfWeek;
		if (fracOfSec < IPS_EPSILON)
		{
			fracOfSec = 0.0;
		}
	}

	tagGPSTIME(int week, int isec, double fsec)
	{
		GPSWeek = week;
		secsOfWeek = isec;
		fracOfSec = fsec;
		if (fracOfSec < IPS_EPSILON)
		{
			fracOfSec = 0.0;
		}
	}

	void operator () (double sec)
	{
		secsOfWeek = int(sec);
		fracOfSec = sec - secsOfWeek;
		if (fracOfSec < IPS_EPSILON)
		{
			fracOfSec = 0.0;
		}
	}

	double operator-(tagGPSTIME gt) ///< GPS时相减
	{
		double lsec = (GPSWeek - gt.GPSWeek)*604800.0 + secsOfWeek - gt.secsOfWeek;
		double t = lsec + fracOfSec - gt.fracOfSec;
		return t;
	}

	tagGPSTIME operator-(double sec) ///< GPS时减去秒数
	{
		tagGPSTIME gt;

		double t = secsOfWeek + fracOfSec - sec;

		long   lt = int(t);
		double st = t - lt;

		double dt = (secsOfWeek - lt) + fracOfSec - sec;

		if (t >= 0 && t < 604800)
		{
			gt.GPSWeek = GPSWeek;
			gt.secsOfWeek = lt;
			gt.fracOfSec = dt;
		}
		else if (t < 0)
		{
			int i = (int)(-lt / 604800) + 1;
			gt.GPSWeek = GPSWeek - i;
			if (dt == 0)
			{
				gt.secsOfWeek = i * 604800 + lt;
				gt.fracOfSec = dt;
			}
			else
			{
				gt.secsOfWeek = i * 604800 + lt - 1;
				gt.fracOfSec = dt + 1;
			}
		}
		else if (t >= 604800)
		{
			int i = (int)(lt / 604800);
			gt.GPSWeek = GPSWeek + i;
			gt.secsOfWeek = lt - i * 604800;
			gt.fracOfSec = dt;
		}

		if (gt.fracOfSec < IPS_EPSILON)
		{
			gt.fracOfSec = 0.0;
		}

		return gt;
	}

	tagGPSTIME operator+(double t) ///< GPS时加上秒数
	{
		double dtmp = 0 - t;

		return (*this - dtmp);
	}

	void operator -= (double t)
	{
		*this = (*this - t);
	}

	void operator += (double t)
	{
		*this = (*this + t);
	}


	bool operator < (tagGPSTIME gt)
	{
		if (*this - gt < 0)
			return true;
		else
			return false;
	}

	bool operator > (tagGPSTIME gt)
	{
		if (*this - gt > 0)
			return true;
		else
			return false;
	}

	bool operator == (tagGPSTIME gt)
	{
		double t = fabs(*this - gt);

		if (isEqual(t, 0))
		{
			return true;
		}
		else
		{
			return false;
		}
	}

	bool operator != (tagGPSTIME gt)
	{
		if (gt == (*this))
			return false;
		else
			return true;
	}

	bool operator <= (tagGPSTIME gt)
	{
		if (*this < gt || *this == gt)
			return true;
		else
			return false;
	}

	bool operator >= (tagGPSTIME gt)
	{
		if (*this > gt || *this == gt)
			return true;
		else
			return false;
	}

#ifndef _WINDOWS
	tagGPSTIME operator-(timespec ts) ///< GPS时减去PC本地时
	{
		double dtmp = ts.tv_sec + ts.tv_nsec*1.0e-9;

		return (*this - dtmp);
	}

	tagGPSTIME operator+(struct timespec ts)///< GPS时加上PC本地时
	{
		double dtmp = 0 - (ts.tv_sec + ts.tv_nsec*1.0e-9);

		return (*this - dtmp);
	}
#endif

} GPSTIME;



typedef struct tagMJD
{
	long        mjd;
	double      fracOfDay;

	tagMJD()
	{
		mjd = 0;
		fracOfDay = 0.0;
	}

}  MJD;

typedef struct tagYDOYHMS
{
	int         year;
	int         yday; ///< dayOfYear
	int         hour;
	int         min;
	double      sec;

	tagYDOYHMS()
	{
		year = yday = hour = min = 0;
		sec = 0.0;
	}

}  YDOYHMS;

typedef struct tagYMDHMS
{
	int         year;
	int         month;
	int         day;
	int         hour;
	int         min;
	double      sec;

	tagYMDHMS()
	{
		year = 2000;
		month = day = 1;
		hour = min = 0;
		sec = 0.0;
	}


	tagYMDHMS(int y, int m, int d, int h, int n, double s)
	{
		year = y;
		month = m;
		day = d;
		hour = h;
		min = n;
		sec = s;
	}


	tagYMDHMS(const double *ep)
	{
		year = (int)ep[0];
		month = (int)ep[1];
		day = (int)ep[2];
		hour = (int)ep[3];
		min = (int)ep[4];
		sec = ep[5];
	}

	///< 简单的减法,精确的还得改
	int operator-(const tagYMDHMS t) const
	{
		int d = (year - t.year) * 365 + (month - t.month) * 30 + (day - t.day);
		return d;
	}

}  YMDHMS;

static const YMDHMS gs_ep2000(2000, 1, 1, 12, 0, 0.0);


typedef struct tagSTRFMT  ///< stream format
{
	int id;
	bool bdefault;
	char head[32];
	char unit[32];
	char format[16];
	int  ffw;             ///< fixed field width
	int  nod;             ///< number of decimals

}STRFMT;



typedef struct tagSOLRPT  ///< solutions report
{
	char   name[1024];    ///< opt文件名
	double Q12_rate;      ///< Q1和Q2所占的百分比
	double Q056_rate;     ///< Q0,Q5,Q6所占的百分比
	double fix_rate;      ///< 固定率
	double good_rate;     ///< 成功解算率
	double rms[2];        ///< rms,0:phase,1:code
	double sig[2];        ///< sig,0:phase,1:code
	double PDOP;          ///< PDOP均值
	long   nepoch;        ///< 历元总数

	tagSOLRPT()
	{
		ZeroStruct(*this, tagSOLRPT);
	}

}SOLRPT;



/*============================ end struct ========================================*/





/*============================ start struct ======================================*/


	///< 具体输出项目
	static const int ExpOptionsNum = 8;			///< 输出设置的数量
	typedef struct tagExpItem
	{
		string ItemName;			///< 显示名称（给用户看的输出条目名称）
		string ExpName;				///< 输出名称（输出到文件中的名称）
		string Formats;				///< 输出格式，如:Degree 
		string FormatsUnit;			///< 输出格式的单位，如:(deg)
		int Options[8];             ///< 输出设置，0:字符类型 1:数据所占字节数 2:小数位字节数 
									///< 3:是否空位填充零 4:是否指数形式输出 5:是否左对齐 6:是否输出小数点后的东西 7:是否用双引号包裹
		tagExpItem()
		{
			ItemName = "";
			ExpName = "";
			Formats = "";
			FormatsUnit = "";
			for (int i = 0; i < ExpOptionsNum; i++) Options[i] = 0;
		}

	} Item;

	///< 输出设置信息
	typedef struct tagExportSetting
	{
		bool bTimeSpan;
		bool bdt;
		bool bds;
		bool bQuality;
		bool bLeverArm;
		bool bRotAngle;

		vector<GPSTIME> TimeSpan[2];///< 输出多段Time[0]到Time[1]之间的结果
		double  dt;					///< 输出一定时间间隔的结果
		double  ds;					///< 输出一定里程间隔的结果
		int		Quality;			///< 输出小于该Quality的结果
		double  LeverArm[3];		///< 某传感器中心相对于惯导的杆臂
		double  RotAngle[3];		///< 惯导安置角，单位为弧度

		vector<Item> ExpItem;	///< 所要输出的条目

		tagExportSetting()
		{
			bTimeSpan = false;
			bdt = false;
			bds = false;
			bQuality = false;
			bLeverArm = false;
			bRotAngle = false;

			dt = 0.0;
			ds = 0.0;
			Quality = 0;
			for (int i = 0; i < 3; i++)
			{
				LeverArm[i] = 0.0;
				RotAngle[i] = 0.0;
			}
		}
	} ExportSetting;

	///< 保存Mark信息
	typedef struct tagEventMark
	{
		GPSTIME  gt;
		int      MarkNum;  // 片号

		tagEventMark()
		{
			MarkNum = -1;
		}

	} EventMark;


	///< ToDo(IPS2.3):
	// 1.存在整型常量的溢出的警告，但这个没什么危害，只是因为gs_prime和gs_basis比较大，导致"Project Name"_hash等比较大
	// unsigned long long为2^64大概为10E19，"Project Name"_hash等很可能超过10E19，溢出就是超过了10E19的表示范围，此时会给溢出的整型重新算个值的
	// 只要保证"str1"_hash和"str2"_hash不重复，溢出是没有危害的
	// 2.如果在Linux下这种方法不可用（需要C++11标准），就改用枚举来实现string的switch/case，之后绝对不能用if/else，特别是在循环中

	/**
	* @brief       定义一个哈希函数，计算出字符串的hash值，将字符串转换为1个整数
	* @note
	*/
	typedef std::uint64_t hash_t;
	static constexpr uint64_t gs_prime = 0x100000001B3ull;
	static constexpr uint64_t gs_basis = 0xCBF29CE484222325ull;

	/**
	* @brief       将字符串转换为整型
	* @param[in]   str      char*
	* @return      hash_t	uint64_t
	* @note
	* @par History:
	*              2021/08/25,Feng Zhu, new \n
	* @internals
	*/
	static constexpr hash_t hash_compile(char const* str, hash_t last_value = gs_basis)
	{
		return *str ? hash_compile(str + 1, (*str ^ last_value) * gs_prime) : last_value;
	};

	/**
	* @brief       定义hash_compile_time的简易表达
	* @param[in]   str      char*
	* @return      hash_t	uint64_t
	* @note
	* @par History:
	*              2021/08/25,Feng Zhu, new \n
	* @internals
	*/
	static constexpr hash_t operator "" _hash(char const* p, size_t)
	{
		return hash_compile(p);
	};


	///< 输出条目
	static const int ExpItemNum = 144;
	static constexpr int ExpItemTypeStartNum[12] = { 0,3,7,18,25,-1,26,36,43,64,94,135 };
	static constexpr hash_t IPSExpItem[ExpItemNum] = {
		// 1.Station Names/Info Range:0-2
		"Project Name"_hash,
		"Sequence Number"_hash,
		"Event Mark Number"_hash,
		// 2.Time/Date Range:3-6
		"GPS Date"_hash,
		"GPS Time"_hash,
		"UTC Date"_hash,
		"UTC Time"_hash,
		// 3.Geographic Coordinates Range:7-17
		"Absolute ECEF-X"_hash,
		"Absolute ECEF-Y"_hash,
		"Absolute ECEF-Z"_hash,
		"Latitude"_hash,
		"Longitude"_hash,
		"GPS Clock"_hash,
		"GLO Clock"_hash,
		"BD2 Clock"_hash,
		"BD3 Clock"_hash,
		"GLA Clock"_hash,
		"QZS Clock"_hash,
		// 4.Grid/Map Coordinates Range:18-24
		"Gauss-North"_hash,
		"Gauss-East"_hash,
		"Grid-North"_hash,
		"Grid-East"_hash,
		"Grid-Up"_hash,
		"UTM-North"_hash,
		"UTM-East"_hash,
		// 5.Height/Elevation Range:25
		"Ellipsoidal Height"_hash,
		// 6.Local Coordinates Range:None(-1表示没有)
		// 7.Velocity/Offsets Range:26-35
		"ECEF Vector X"_hash,
		"ECEF Vector Y"_hash,
		"ECEF Vector Z"_hash,
		"East Velocity"_hash,
		"North Velocity"_hash,
		"Up Velocity"_hash,
		"Body Frame X-Velocity"_hash,
		"Body Frame Y-Velocity"_hash,
		"Body Frame Z-Velocity"_hash,
		"Horizontal Velocity"_hash,
		// 8.Baseline Information Range:36-42
		"BL Vector dE"_hash,
		"BL Vector dN"_hash,
		"BL Vector dU"_hash,
		"Relative Slope Horizontal Distance"_hash,
		"Total Slope Horizontal Distance"_hash,
		"Relative Slope Distance"_hash,
		"Total Slope Distance"_hash,
		// 9.Standard Deviation Range:43-63
		"East StdDev"_hash,
		"North StdDev"_hash,
		"Up StdDev"_hash,
		"ECEF X StdDev"_hash,
		"ECEF Y StdDev"_hash,
		"ECEF Z StdDev"_hash,
		"East Velocity StdDev"_hash,
		"North Velocity StdDev"_hash,
		"Up Velocity StdDev"_hash,
		"ECEF X Velocity StdDev"_hash,
		"ECEF Y Velocity StdDev"_hash,
		"ECEF Z Velocity StdDev"_hash,
		"Heading StdDev"_hash,
		"Pitch StdDev"_hash,
		"Roll StdDev"_hash,
		"GPS Clock StdDev"_hash,
		"GLO Clock StdDev"_hash,
		"BD2 Clock StdDev"_hash,
		"BD3 Clock StdDev"_hash,
		"GLA Clock StdDev"_hash,
		"QZS Clock StdDev"_hash,
		// 10.Covariance Range:64-93
		"ECEF Covariance Cx(1,1)"_hash,
		"ECEF Covariance Cx(2,2)"_hash,
		"ECEF Covariance Cx(3,3)"_hash,
		"ECEF Covariance Cx(1,2)"_hash,
		"ECEF Covariance Cx(1,3)"_hash,
		"ECEF Covariance Cx(2,3)"_hash,
		"Local Covariance Cx(1,1)"_hash,
		"Local Covariance Cx(2,2)"_hash,
		"Local Covariance Cx(3,3)"_hash,
		"Local Covariance Cx(1,2)"_hash,
		"Local Covariance Cx(1,3)"_hash,
		"Local Covariance Cx(2,3)"_hash,
		"ECEF Velocity Covariance Cx(1,1)"_hash,
		"ECEF Velocity Covariance Cx(2,2)"_hash,
		"ECEF Velocity Covariance Cx(3,3)"_hash,
		"ECEF Velocity Covariance Cx(1,2)"_hash,
		"ECEF Velocity Covariance Cx(1,3)"_hash,
		"ECEF Velocity Covariance Cx(2,3)"_hash,
		"Local Velocity Covariance Cx(1,1)"_hash,
		"Local Velocity Covariance Cx(2,2)"_hash,
		"Local Velocity Covariance Cx(3,3)"_hash,
		"Local Velocity Covariance Cx(1,2)"_hash,
		"Local Velocity Covariance Cx(1,3)"_hash,
		"Local Velocity Covariance Cx(2,3)"_hash,
		"Att Heading Cov.CHH"_hash,
		"Att Pitch Cov.CPP"_hash,
		"Att Roll Cov.CRR"_hash,
		"Att HeadingPitch Cov.CHP"_hash,
		"Att HeadingRoll Cov.CHR"_hash,
		"Att PitchRoll Cov.CPR"_hash,
		// 11.Statistics Range:94-132
		"Ratio"_hash,
		"Ambiguity Status"_hash,
		"IMU Update Types"_hash,
		"East Separation"_hash,
		"North Separation"_hash,
		"Up Separation"_hash,
		"Vel East Separation"_hash,
		"Vel North Separation"_hash,
		"Vel Up Separation"_hash,
		"Heading Separation"_hash,
		"Pitch Separation"_hash,
		"Roll Separation"_hash,
		"RMS-L"_hash,
		"Sigma-L"_hash,
		"RMS-C/A"_hash,
		"Sigma-C/A"_hash,
		"Number of Satellites"_hash,
		"Fixed Number of Satellites"_hash,
		"Visible GPS Num"_hash,
		"Solved GPS Num"_hash,
		"Fixed GPS Num"_hash,
		"Visible GLO Num"_hash,
		"Solved GLO Num"_hash,
		"Fixed GLO Num"_hash,
		"Visible BD2 Num"_hash,
		"Solved BD2 Num"_hash,
		"Fixed BD2 Num"_hash,
		"Visible BD3 Num"_hash,
		"Solved BD3 Num"_hash,
		"Fixed BD3 Num"_hash,
		"Visible GAL Num"_hash,
		"Solved GAL Num"_hash,
		"Fixed GAL Num"_hash,
		"Visible QZS Num"_hash,
		"Solved QZS Num"_hash,
		"Fixed QZS Num"_hash,
		"Double Difference DOP"_hash,
		"PDOP"_hash,
		"HDOP"_hash,
		"VDOP"_hash,
		"Quality Number"_hash,
		// 12.IMU/Attitude Parameters Range:135-143
		"Heading Angle"_hash,
		"Pitch Angle"_hash,
		"Roll Angle"_hash,
		"AccelBias-X"_hash,
		"GyroBias-X"_hash,
		"AccelBias-Y"_hash,
		"GyroBias-Y"_hash,
		"AccelBias-Z"_hash,
		"GyroBias-Z"_hash
	};

	///< 一般量的输出格式
	static const int ExpValueFormatNum = 9;
	static constexpr hash_t IPSExpValueFormat[ExpValueFormatNum] = {
		"Metres"_hash,
		"Centimeters"_hash,
		"Millimeters"_hash,
		"Kilometers"_hash,
		"Meters per Second"_hash,
		"Kilometers per Hour"_hash,
		"Meters per Second Squared"_hash,
		"Milligravity"_hash,
		"Degrees per Hour"_hash
	};

	///< 角度相关量的输出格式
	static const int ExpDegreeFormatNum = 11;
	static constexpr hash_t IPSExpDegreeFormat[ExpDegreeFormatNum] = {
		"Degrees Minutes(signed)"_hash,
		"Degrees Minutes(no sign)"_hash,
		"Degrees-Minutes(no sign)"_hash,
		"Degrees Minutes Sec(signed)"_hash,
		"Degrees Minutes Sec(no sign)"_hash,
		"Degrees-Minutes-Sec(no sign)"_hash,
		"Decimal Degrees(signed)"_hash,
		"Decimal Degrees(no sign)"_hash,
		"Decimal Degrees(no sign.0-360)"_hash,
		"Decimal Minutes(0-60)"_hash,
		"Decimal Seconds(0-60)"_hash
	};

	///< 时间相关量的输出格式
	static const int ExpTimeFormatNum = 18;
	static constexpr hash_t IPSExpTimeFormat[ExpTimeFormatNum] = {
		"Seconds Of the Week"_hash,
		"HH:MM:SS.SS"_hash,
		"Hours"_hash,
		"Minutes"_hash,
		"Seconds"_hash,
		"GPS Weeks"_hash,
		"Seconds of the Day"_hash,
		"Seconds since 1980"_hash,
		"Seconds since 1970"_hash,
		"Month/Day/Year"_hash,
		"Day/Month/Year"_hash,
		"Day.Month.Year"_hash,
		"Year/Month/Day"_hash,
		"Year(4 digits)"_hash,
		"Year(2 digits)"_hash,
		"Month(1-12)"_hash,
		"Day of Month(1-31)"_hash,
		"Day of Week(0-6)"_hash
	};


	///< OPTION
	static const int GNSSOptNum = 173;
	static constexpr int GNSSOptItemStartNum[11] = { 0,2,27,43,53,71,86,114,122,128,138 };
	static constexpr hash_t GNSSOptItem[GNSSOptNum] = {
		// 0.POSMind界面
		"PROJ_PATH"_hash,
		"PROJ_NAME"_hash,
		// 1.GNSS输入文件
		"GNSS_ResultPath"_hash,
		"GNSS_FilePath"_hash,
		"GNSS_OBSFile"_hash,
		"GNSS_ALLNavFile"_hash,
		"GNSS_GPSNavFile"_hash,
		"GNSS_GLONavFile"_hash,
		"GNSS_BD2NavFile"_hash,
		"GNSS_BD3NavFile"_hash,
		"GNSS_GALNavFile"_hash,
		"GNSS_QZSNavFile"_hash,
		"GNSS_GPSPreEphFile"_hash,
		"GNSS_GPSPreClkFile"_hash,
		"GNSS_GLOPreEphFile"_hash,
		"GNSS_GLOPreClkFile"_hash,
		"GNSS_BD2PreEphFile"_hash,
		"GNSS_BD2PreClkFile"_hash,
		"GNSS_BD3PreEphFile"_hash,
		"GNSS_BD3PreClkFile"_hash,
		"GNSS_GALPreEphFile"_hash,
		"GNSS_GALPreClkFile"_hash,
		"GNSS_QZSPreEphFile"_hash,
		"GNSS_QZSPreClkFile"_hash,
		"GNSS_PPPAR_FCBFile"_hash,
		"GNSS_MarkFile"_hash,
		"GNSS_ExpXMLFile"_hash,
		// 2.GNSS处理方式
		"GNSS_bGNSSProcAll"_hash,
		"GNSS_BeginTime"_hash,
		"GNSS_EndTime"_hash,
		"GNSS_ProcRate"_hash,
		"GNSS_PMode"_hash,
		"GNSS_FilterType"_hash,
		"GNSS_PPPType"_hash,
		"GNSS_RTKType"_hash,
		"GNSS_UserType"_hash,
		"GNSS_SceneType"_hash,
		"GNSS_SatSYS"_hash,
		"GNSS_BaseXYZVRSFile"_hash,
		"GNSS_BaseXYZCoorSys"_hash,
		"GNSS_BaseXYZ"_hash,
		"GNSS_AntMode"_hash,
		"GNSS_AntHeight"_hash,
		// 3.GNSS频点和码通道设置
		"GNSS_bFreqUsed"_hash,
		"GNSS_bFreqUsedSPP"_hash,
		"GNSS_bSwitchGNSSFreq"_hash,
		"GNSS_GPSFreq"_hash,
		"GNSS_GLOFreq"_hash,
		"GNSS_BD2Freq"_hash,
		"GNSS_BD3Freq"_hash,
		"GNSS_GALFreq"_hash,
		"GNSS_QZSFreq"_hash,
		"GNSS_CodePrior"_hash,
		// 4.GNSS预处理参数
		"GNSS_ElevMask"_hash,
		"GNSS_ErrorPLR"_hash,
		"GNSS_Error_a"_hash,
		"GNSS_Error_b"_hash,
		"GNSS_Error_ppm"_hash,
		"GNSS_sClkStab"_hash,
		"GNSS_ErrDop_a"_hash,
		"GNSS_ErrDop_b"_hash,
		"GNSS_ErrDop_ppm"_hash,
		"GNSS_Error_EF"_hash,
		"GNSS_CSRMode"_hash,
		"GNSS_bCS_LLI"_hash,
		"GNSS_bCS_GF"_hash,
		"GNSS_bCS_MW"_hash,
		"GNSS_CSGFThres"_hash,
		"GNSS_CSMWThres"_hash,
		"GNSS_bCJ"_hash,
		"GNSS_bExcludedPRN"_hash,
		// 5.GNSS误差改正
		"GNSS_ATXFile"_hash,
		"GNSS_DCBFile"_hash,
		"GNSS_ERPFile"_hash,
		"GNSS_BLQFile"_hash,
		"GNSS_IONFile"_hash,
		"GNSS_bTideSolidCorr"_hash,
		"GNSS_bTideOceanCorr"_hash,
		"GNSS_bTidePoleCorr"_hash,
		"GNSS_bTideAtmosphereCorr"_hash,
		"GNSS_bSatPCVCorr"_hash,
		"GNSS_bSatPCOCorr"_hash,
		"GNSS_bRcvPCVCorr"_hash,
		"GNSS_bRcvPCOCorr"_hash,
		"GNSS_bWindupCorr"_hash,
		"GNSS_bTestEclipse"_hash,
		// 6.GNSS模糊度固定
		"GNSS_ARMode"_hash,
		"GNSS_ARMethod"_hash,
		"GNSS_ARType"_hash,
		"GNSS_bPAR"_hash,
		"GNSS_ARWideLaneType"_hash,
		"GNSS_ARThres"_hash,
		"GNSS_ARFFRTThres"_hash,
		"GNSS_ARElevMask"_hash,
		"GNSS_ARMinLock"_hash,
		"GNSS_ARAmbLock"_hash,
		"GNSS_ARAmbFrac"_hash,
		"GNSS_ARAmbSig"_hash,
		"GNSS_ARAmbSR"_hash,
		"GNSS_ARnSat"_hash,
		"GNSS_ARADOP"_hash,
		"GNSS_ARSRBS"_hash,
		"GNSS_ARHold_ElevMask"_hash,
		"GNSS_ARHold_nFixEpoch"_hash,
		"GNSS_ARHold_nSat"_hash,
		"GNSS_ARHold_ADOP"_hash,
		"GNSS_ARHold_SRBS"_hash,
		"GNSS_ARHold_Ratio"_hash,
		"GNSS_ARHold_nEpoch"_hash,
		"GNSS_ARHold_Sig"_hash,
		"GNSS_ARCLambdaCons"_hash,
		"GNSS_ARCLambdaConsSig"_hash,
		"GNSS_ARVLambdaCons"_hash,
		"GNSS_ARVLambdaConsSig"_hash,
		// 7.GNSS大气误差修正
		"GNSS_TropModel"_hash,
		"GNSS_TropMF"_hash,
		"GNSS_bAddTropGrad"_hash,
		"GNSS_TropGrad"_hash,
		"GNSS_TropEstType"_hash,
		"GNSS_IonoModel"_hash,
		"GNSS_IonoModelSPP"_hash,
		"GNSS_IonoEstType"_hash,
		// 8.GNSS数据质量控制
		"GNSS_bQualityControl"_hash,
		"GNSS_DOPThres"_hash,
		"GNSS_bRAIM_FED"_hash,
		"GNSS_DataGapTol"_hash,
		"GNSS_MaxOutage"_hash,
		"GNSS_GNSSMaxInno"_hash,
		// 9.GNSS参数估计策略
		"GNSS_SPPEph"_hash,
		"GNSS_bSPPEKF"_hash,
		"GNSS_bSPPRTD"_hash,
		"GNSS_PPPEph"_hash,
		"GNSS_RTKEph"_hash,
		"GNSS_PosEstMode"_hash,
		"GNSS_ClkEstMode"_hash,
		"GNSS_bDerivedDoppler"_hash,
		"GNSS_bUseDoppler"_hash,
		"GNSS_DistAlpha"_hash,
		// 10.GNSS误差模型参数
		"GNSS_BASE_ISDV_Pos"_hash,
		"GNSS_BASE_PNSD_Pos"_hash,
		"GNSS_BASE_ISDV_Vel"_hash,
		"GNSS_BASE_PNSD_Vel"_hash,
		"GNSS_BASE_ISDV_Acc"_hash,
		"GNSS_BASE_PNSD_Acc"_hash,
		"GNSS_BASE_ISDV_Clk"_hash,
		"GNSS_BASE_PNSD_Clk"_hash,
		"GNSS_BASE_ISDV_ClkVel"_hash,
		"GNSS_BASE_PNSD_ClkVel"_hash,
		"GNSS_BASE_ISDV_ZWD"_hash,
		"GNSS_BASE_PNSD_ZWD"_hash,
		"GNSS_BASE_ISDV_GradNorth"_hash,
		"GNSS_BASE_PNSD_GradNorth"_hash,
		"GNSS_BASE_ISDV_GradEast"_hash,
		"GNSS_BASE_PNSD_GradEast"_hash,
		"GNSS_BASE_ISDV_Ion"_hash,
		"GNSS_BASE_PNSD_Ion"_hash,
		"GNSS_BASE_ISDV_PosSPP"_hash,
		"GNSS_BASE_ISDV_ClkSPP"_hash,
		"GNSS_BASE_ISDV_ClkVelSPP"_hash,
		"GNSS_PPP_ISDV_Amb"_hash,
		"GNSS_PPP_PNSD_GPSAmb"_hash,
		"GNSS_PPP_PNSD_GLOAmb"_hash,
		"GNSS_PPP_PNSD_BD2Amb"_hash,
		"GNSS_PPP_PNSD_BD3Amb"_hash,
		"GNSS_PPP_PNSD_GALAmb"_hash,
		"GNSS_PPP_PNSD_QZSAmb"_hash,
		"GNSS_RTK_ISDV_Amb"_hash,
		"GNSS_RTK_PNSD_GPSAmb"_hash,
		"GNSS_RTK_PNSD_GLOAmb"_hash,
		"GNSS_RTK_PNSD_BD2Amb"_hash,
		"GNSS_RTK_PNSD_BD3Amb"_hash,
		"GNSS_RTK_PNSD_GALAmb"_hash,
		"GNSS_RTK_PNSD_QZSAmb"_hash
	};

	static const int SINSOptNum = 103;
	static constexpr int SINSOptItemStartNum[9] = { 0,13,20,33,42,46,61,76,92 };
	static constexpr hash_t SINSOptItem[SINSOptNum] = {
		// 0.SINS输入文件
		"SINS_ResultPath"_hash,
		"SINS_FilePath"_hash,
		"SINS_IMUFile"_hash,
		"SINS_GNSSFile"_hash,
		"SINS_POSFile"_hash,
		"SINS_DMIFile"_hash,
		"SINS_HEDFile"_hash,
		"SINS_ZUPTFile"_hash,
		"SINS_ZIHRFile"_hash,
		"SINS_NHCFile"_hash,
		"SINS_CUPTFile"_hash,
		"SINS_MarkFile"_hash,
		"SINS_ExpXMLFile"_hash,
		// 1.SINS处理方式
		"SINS_NaviSys"_hash,
		"SINS_PMode"_hash,
		"SINS_FilterType"_hash,
		"SINS_bRTS"_hash,
		"SINS_bIMUProcAll"_hash,
		"SINS_IMUBeginTime"_hash,
		"SINS_IMUEndTime"_hash,
		// 2.SINS基本设置
		"SINS_InitAtt"_hash,
		"SINS_InitVel"_hash,
		"SINS_InitPos"_hash,
		"SINS_bStateUPTDeltTFromIMU"_hash,
		"SINS_StateUPTDeltT"_hash,
		"SINS_bAzimuth"_hash,
		"SINS_bAddAccelBiasScale"_hash,
		"SINS_bAddAccelBiasTob"_hash,
		"SINS_bAddAccelBiasOrthog"_hash,
		"SINS_bAddGyrosBiasScale"_hash,
		"SINS_bAddGyrosBiasTob"_hash,
		"SINS_bAddGyrosBiasOrthog"_hash,
		"SINS_IMUDataGapTol"_hash,
		// 3.SINS安装参数
		"SINS_LeverArmbIMU"_hash,
		"SINS_RotAngleIMU"_hash,
		"SINS_bEstLeverArmbGNSS"_hash,
		"SINS_LeverArmbGNSS"_hash,
		"SINS_LeverArmDMI"_hash,
		"SINS_bEstWheelSizeDMI"_hash,
		"SINS_WheelSizeDMI"_hash,
		"SINS_bEstRotAngleNHC"_hash,
		"SINS_RotAngleNHC"_hash,
		// 4.SINS初始对准
		"SINS_AlignmentMode"_hash,
		"SINS_SCATotalSec"_hash,
		"SINS_SFATotalSec"_hash,
		"SINS_SATimeSpanProc"_hash,
		// 5.SINS多源信息更新
		"SINS_bPOSUsed"_hash,
		"SINS_bVELUsed"_hash,
		"SINS_bATTUsed"_hash,
		"SINS_bDMIUsed"_hash,
		"SINS_bHEDUsed"_hash,
		"SINS_bZUPUsed"_hash,
		"SINS_bZIHUsed"_hash,
		"SINS_bCUPUsed"_hash,
		"SINS_bPHSUsed"_hash,
		"SINS_bCODUsed"_hash,
		"SINS_bDOPUsed"_hash,
		"SINS_bDISUsed"_hash,
		"SINS_bNHCUsed"_hash,
		"SINS_bHGHUsed"_hash,
		"SINS_bLANUsed"_hash,
		// 6.SINS多源信息观测更新率
		"SINS_POSRate"_hash,
		"SINS_VELRate"_hash,
		"SINS_ATTRate"_hash,
		"SINS_DMIRate"_hash,
		"SINS_HEDRate"_hash,
		"SINS_ZUPRate"_hash,
		"SINS_ZIHRate"_hash,
		"SINS_CUPRate"_hash,
		"SINS_PHSRate"_hash,
		"SINS_CODRate"_hash,
		"SINS_DOPRate"_hash,
		"SINS_DISRate"_hash,
		"SINS_NHCRate"_hash,
		"SINS_HGHRate"_hash,
		"SINS_LANRate"_hash,
		// 7.SINS多源信息质量控制
		"SINS_bQualityControl"_hash,
		"SINS_POSSigThres"_hash,
		"SINS_VELSigThres"_hash,
		"SINS_ATTSigThres"_hash,
		"SINS_DMISigThres"_hash,
		"SINS_HEDSigThres"_hash,
		"SINS_ZUPSigThres"_hash,
		"SINS_ZIHSigThres"_hash,
		"SINS_CUPSigThres"_hash,
		"SINS_PHSSigThres"_hash,
		"SINS_CODSigThres"_hash,
		"SINS_DOPSigThres"_hash,
		"SINS_DISSigThres"_hash,
		"SINS_NHCSigThres"_hash,
		"SINS_HGHSigThres"_hash,
		"SINS_LANSigThres"_hash,
		// 8.SINS误差模型参数
		"SINS_IMUType"_hash,
		"SINS_ISDV_LeverArmbGNSS"_hash,
		"SINS_PNSD_LeverArmbGNSS"_hash,
		"SINS_MSSD_DMI"_hash,
		"SINS_ISDV_WheelSizeDMI"_hash,
		"SINS_PNSD_WheelSizeDMI"_hash,
		"SINS_MSSD_NHC"_hash,
		"SINS_ISDV_RotAngleNHC"_hash,
		"SINS_PNSD_RotAngleNHC"_hash,
		"SINS_MSSD_ZUP"_hash,
		"SINS_MSSD_ZIH"_hash
	};

/*============================ end struct ========================================*/

}

#endif
