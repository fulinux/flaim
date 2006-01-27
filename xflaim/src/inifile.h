//------------------------------------------------------------------------------
// Desc: Class to support reading/writing/parsing of .ini file
//			Every line in the .ini file may have a parameter part (consisting of a 
//			parameter name and an optional value) and a comment part. Empty lines
//			are also allowed.  Parameter names don't have to have values, but a
//			value must have a name.
//
// Tabs:	3
//
//		Copyright (c) 2002-2006 Novell, Inc. All Rights Reserved.
//
//		This program is free software; you can redistribute it and/or
//		modify it under the terms of version 2 of the GNU General Public
//		License as published by the Free Software Foundation.
//
//		This program is distributed in the hope that it will be useful,
//		but WITHOUT ANY WARRANTY; without even the implied warranty of
//		MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
//		GNU General Public License for more details.
//
//		You should have received a copy of the GNU General Public License
//		along with this program; if not, contact Novell, Inc.
//
//		To contact Novell about this file by physical or electronic mail,
//		you may find current contact information at www.novell.com
//
// $Id: inifile.h 3109 2006-01-19 13:07:07 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------

typedef struct IniLine
{
	char *				pszParamName;
	char *				pszParamValue;	
	char *				pszComment;
	struct IniLine *	pPrev;
	struct IniLine *	pNext;
} INI_LINE;

class F_IniFile : public XF_RefCount, public XF_Base
{
public:

	F_IniFile();
	
	virtual ~F_IniFile();
	
	RCODE Init( void);
	
	RCODE Read(
		char *			pszFileName);
		
	RCODE Write( void);

	FLMBOOL GetParam(
		const char *	pszParamName,
		FLMUINT *		puiParamVal);
	
	RCODE SetParam(
		const char *	pszParamName,
		FLMUINT 			uiParamVal);

	FLMBOOL GetParam(
		const char *	pszParamName,
		FLMBOOL *		pbParamVal);
	
	RCODE SetParam(
		const char *	pszParamName,
		FLMBOOL			bParamVal);

	FLMBOOL GetParam(
		const char *	pszParamName,
		char **			ppszParamVal);
	
	RCODE SetParam(
		const char *	pszParamName,
		const char *	pszParamVal);

	FLMBOOL TestParam(
		const char *	pszParamName)
	{
		if( findParam( pszParamName))
		{
			return( TRUE);
		}
		
		return( FALSE);
	}

private:

	RCODE readLine(
		char *			pucBuf,
		FLMUINT *		puiBytes,
		FLMBOOL *		pbMore);

	RCODE parseBuffer(
		char *			pucBuf,
		FLMUINT			uiNumButes);

	INI_LINE * findParam(
		const char *	pszParamName);

	RCODE setParamCommon( 
		INI_LINE **		ppLine,
		const char *	pszParamName);

	void fromAscii( 
		FLMUINT * 		puiVal,
		const char *	pszParamValue);
		
	void fromAscii(
		FLMBOOL *		pbVal,
		const char *	pszParamValue);

	RCODE toAscii( 
		char **			ppszParamValue,
		FLMUINT			puiVal);
		
	RCODE toAscii( 
		char **			ppszParamValue,
		FLMBOOL 			pbVal);
		
	RCODE toAscii(
		char **			ppszParamValue,
		const char * 	pszVal);

	FINLINE FLMBOOL isWhiteSpace(
		FLMBYTE			ucChar)
	{
		return( ucChar == 32 || ucChar == 9 ? TRUE : FALSE);
	}
	
	IF_Pool *			m_pPool;
	IF_FileHdl * 		m_pFile;
	char *				m_pszFileName;
	INI_LINE *			m_pFirstLine;	
	INI_LINE *			m_pLastLine;
	FLMBOOL				m_bReady;
	FLMBOOL				m_bModified;
	FLMUINT				m_uiFileOffset;
};
