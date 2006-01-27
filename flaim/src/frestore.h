//-------------------------------------------------------------------------
// Desc:	Database restore - class definitions
// Tabs:	3
//
//		Copyright (c) 2001-2006 Novell, Inc. All Rights Reserved.
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
// $Id: frestore.h 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#ifndef FRESTORE_H
#define FRESTORE_H

#include "fpackon.h"
// IMPORTANT NOTE: No other include files should follow this one except
// for fpackoff.h

/****************************************************************************
Desc:	This object is our implementation of the
		F_UnknownStream object which is used to handle unknown
		objects in the RFL.
****************************************************************************/
class F_RflUnknownStream : public F_UnknownStream
{
public:

	F_RflUnknownStream();
	virtual ~F_RflUnknownStream();

	RCODE setup(
		F_Rfl *			pRfl,
		FLMBOOL			bInputStream);

	RCODE reset( void);

	RCODE read(
		FLMUINT			uiLength,				// Number of bytes to read
		void *			pvBuffer,				// Buffer to place read bytes into
		FLMUINT *		puiBytesRead);			// [out] Number of bytes read

	RCODE write(
		FLMUINT			uiLength,				// Number of bytes to write
		void *			pvBuffer);

	RCODE close( void);							// Reads to the end of the
														// stream and discards any
														// remaining data (if input stream).

private:

	FLMBOOL		m_bSetupCalled;
	F_Rfl *		m_pRfl;					// RFL object.
	FLMBOOL		m_bInputStream;		// TRUE=input stream, FALSE=output stream
	FLMBOOL		m_bStartedWriting;	// Only used for output streams
};

/****************************************************************************
Desc:	The F_FSRestore class is used to read backup and RFL files from 
		a disk file system.
****************************************************************************/
class F_FSRestore : public F_Restore
{
public:

	F_FSRestore();
	
	virtual ~F_FSRestore();

	RCODE setup(
		const char *		pucDbPath,
		const char *		pucBackupSetPath,
		const char *		pucRflDir);

	RCODE openBackupSet( void);

	RCODE openIncFile(
		FLMUINT				uiFileNum);

	RCODE openRflFile(
		FLMUINT			uiFileNum);

	RCODE read(
		FLMUINT			uiLength,
		void *			pvBuffer,
		FLMUINT *		puiBytesRead);

	RCODE close( void);

	FINLINE RCODE abortFile( void)
	{
		return( close());
	}

	FINLINE RCODE processUnknown(
		F_UnknownStream *		pUnkStrm)
	{
		// Skip any unknown data in the RFL

		return( pUnkStrm->close());
	}

	FINLINE RCODE status(
		eRestoreStatusType	eStatusType,
		FLMUINT					uiTransId,
		void *					pvValue1,
		void *					pvValue2,
		void *					pvValue3,
		eRestoreActionType *	peRestoreAction)
	{
		F_UNREFERENCED_PARM( eStatusType);
		F_UNREFERENCED_PARM( uiTransId);
		F_UNREFERENCED_PARM( pvValue1);
		F_UNREFERENCED_PARM( pvValue2);
		F_UNREFERENCED_PARM( pvValue3);

		*peRestoreAction = RESTORE_ACTION_CONTINUE;
		return( FERR_OK);
	}

private:

	F_FileHdlImp *			m_pFileHdl;
	F_64BitFileHandle *	m_pFileHdl64;
	FLMUINT64				m_ui64Offset;
	FLMUINT					m_uiDbVersion;
	char						m_szDbPath[ F_PATH_MAX_SIZE];
	char						m_szBackupSetPath[ F_PATH_MAX_SIZE];
	char						m_szRflDir[ F_PATH_MAX_SIZE];
	FLMBOOL					m_bSetupCalled;
	FLMBOOL					m_bOpen;
};

#include "fpackoff.h"

#endif
