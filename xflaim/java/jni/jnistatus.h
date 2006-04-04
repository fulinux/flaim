//------------------------------------------------------------------------------
// Desc:
//
// Tabs:	3
//
//		Copyright (c) 2003-2006 Novell, Inc. All Rights Reserved.
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
// $Id: $
//------------------------------------------------------------------------------

#include "xflaim.h"
#include "flaimsys.h"
#include <jni.h>

/****************************************************************************
Desc:
****************************************************************************/
class JNIRenameStatus : public IF_DbRenameStatus, public XF_Base
{
public:

	JNIRenameStatus(
		jobject		jStatus,
		JavaVM *		pJvm)
	{
		flmAssert( jStatus);
		flmAssert( pJvm);
		m_jStatus = jStatus;
		m_pJvm = pJvm;
	}

	RCODE XFLMAPI dbRenameStatus(
		const char *	pszSrcFileName,
		const char *	pszDstFileName);

	FINLINE FLMUINT getRefCount( void)
	{
		return( IF_DbRenameStatus::getRefCount());
	}

	virtual FINLINE FLMINT XFLMAPI AddRef( void)
	{
		return( IF_DbRenameStatus::AddRef());
	}

	virtual FINLINE FLMINT XFLMAPI Release( void)
	{
		return( IF_DbRenameStatus::Release());
	}

private:

	JavaVM *		m_pJvm;
	jobject		m_jStatus;
};

/****************************************************************************
Desc:
****************************************************************************/
class JNICopyStatus : public IF_DbCopyStatus, public XF_Base
{
public:

	JNICopyStatus(
		jobject		jStatus,
		JavaVM *		pJvm)
	{
		flmAssert( jStatus);
		flmAssert( pJvm);
		m_jStatus = jStatus;
		m_pJvm = pJvm;
	}
	
	RCODE XFLMAPI dbCopyStatus(
		FLMUINT64		ui64BytesToCopy,
		FLMUINT64		ui64BytesCopied,
		FLMBOOL			bNewSrcFile,
		const char *	pszSrcFileName,
		const char *	pszDestFileName);
		
	FINLINE FLMUINT getRefCount( void)
	{
		return( IF_DbCopyStatus::getRefCount());
	}

	virtual FINLINE FLMINT XFLMAPI AddRef( void)
	{
		return( IF_DbCopyStatus::AddRef());
	}

	virtual FINLINE FLMINT XFLMAPI Release( void)
	{
		return( IF_DbCopyStatus::Release());
	}

private:

	JavaVM *		m_pJvm;
	jobject		m_jStatus;	
};

/****************************************************************************
Desc:
****************************************************************************/
class JNICheckStatus : public IF_DbCheckStatus, public XF_Base
{
public:

	JNICheckStatus(
		jobject		jStatus,
		JavaVM *		pJvm)
	{
		flmAssert( jStatus);
		flmAssert( pJvm);
		m_jStatus = jStatus;
		m_pJvm = pJvm;
	}
	
	RCODE XFLMAPI reportProgress(
		XFLM_PROGRESS_CHECK_INFO *	pProgCheck);

	RCODE XFLMAPI reportCheckErr(
		XFLM_CORRUPT_INFO *	pCorruptInfo,
		FLMBOOL *				pbFix);
		
	FINLINE FLMUINT getRefCount( void)
	{
		return( IF_DbCheckStatus::getRefCount());
	}

	virtual FINLINE FLMINT XFLMAPI AddRef( void)
	{
		return( IF_DbCheckStatus::AddRef());
	}

	virtual FINLINE FLMINT XFLMAPI Release( void)
	{
		return( IF_DbCheckStatus::Release());
	}

private:

	JavaVM *		m_pJvm;
	jobject		m_jStatus;	
};
