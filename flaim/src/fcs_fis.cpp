//-------------------------------------------------------------------------
// Desc:	File input stream class.
// Tabs:	3
//
//		Copyright (c) 2000,2003,2005-2006 Novell, Inc. All Rights Reserved.
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
// $Id: fcs_fis.cpp 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#include "flaimsys.h"

/****************************************************************************
Desc:
*****************************************************************************/
FCS_FIS::FCS_FIS( void)
{
	m_pFileHdl = NULL;
	m_pucBufPos = NULL;
	m_pucBuffer = NULL;
	m_uiFileOffset = 0;
	m_uiBlockSize = 0;
	m_uiBlockEnd = 0;
}

/****************************************************************************
Desc:
*****************************************************************************/
FCS_FIS::~FCS_FIS( void)
{
	if( m_pFileHdl)
	{
		m_pFileHdl->Release();
	}

	if( m_pucBuffer)
	{
		f_free( &m_pucBuffer);
	}
}

/****************************************************************************
Desc:	Configures the input stream for use
*****************************************************************************/
RCODE FCS_FIS::setup(
	const char *	pszFilePath,
	FLMUINT			uiBlockSize)
{
	RCODE			rc = FERR_OK;
	
	flmAssert( uiBlockSize);

	if( RC_BAD( rc = close()))
	{
		goto Exit;
	}

	if( RC_BAD( rc = gv_FlmSysData.pFileSystem->Open( pszFilePath,
		F_IO_RDONLY | F_IO_SH_DENYNONE, &m_pFileHdl)))
	{
		goto Exit;
	}

	m_uiBlockSize = uiBlockSize;
	if( RC_BAD( rc = f_alloc( m_uiBlockSize, &m_pucBuffer)))
	{
		goto Exit;
	}
	m_pucBufPos = m_pucBuffer;

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Closes the input stream and frees any resources
*****************************************************************************/
RCODE FCS_FIS::close( void)
{
	if( m_pFileHdl)
	{
		m_pFileHdl->Close();
		m_pFileHdl->Release();
		m_pFileHdl = NULL;
	}

	if( m_pucBuffer)
	{
		f_free( &m_pucBuffer);
	}
	
	return( FERR_OK);
}

/****************************************************************************
Desc:	Reads the requested amount of data from the stream.
*****************************************************************************/
RCODE FCS_FIS::read(
	FLMBYTE *		pucData,
	FLMUINT			uiLength,
	FLMUINT *		puiBytesRead)
{
	RCODE			rc = FERR_OK;
	FLMUINT		uiBytesRead = 0;
	FLMUINT		uiMaxSize;

	if( puiBytesRead)
	{
		*puiBytesRead = 0;
	}

	if( !m_pFileHdl)
	{
		rc = RC_SET( FERR_READING_FILE);
		goto Exit;
	}

	while( uiLength)
	{
		uiMaxSize = m_uiBlockEnd - (FLMUINT)(m_pucBufPos - m_pucBuffer);

		if( !uiMaxSize)
		{
			if( RC_BAD( rc = getNextPacket()))
			{
				goto Exit;
			}
		}
		else if( uiLength <= uiMaxSize)
		{
			f_memcpy( pucData, m_pucBufPos, uiLength);
			m_pucBufPos += uiLength;
			uiBytesRead += uiLength;
			uiLength = 0;
		}
		else
		{
			f_memcpy( pucData, m_pucBufPos, uiMaxSize);
			m_pucBufPos += uiMaxSize;
			pucData += uiMaxSize;
			uiLength -= uiMaxSize;
			uiBytesRead += uiMaxSize;
		}
	}

Exit:

	if( puiBytesRead)
	{
		*puiBytesRead = uiBytesRead;
	}

	return( rc);
}

/****************************************************************************
Desc:	Flushes any pending data.
*****************************************************************************/
RCODE FCS_FIS::flush( void)
{
	return( FERR_OK);
}

/****************************************************************************
Desc:	Flushes any pending data.
*****************************************************************************/
RCODE FCS_FIS::endMessage( void)
{
	return( FERR_OK);
}

/****************************************************************************
Desc:	Returns TRUE if the stream is open
*****************************************************************************/
FLMBOOL FCS_FIS::isOpen( void)
{
	return( TRUE);
}

/****************************************************************************
Desc:	Reads the next block from the file
*****************************************************************************/
RCODE FCS_FIS::getNextPacket( void)
{
	RCODE			rc = FERR_OK;

	if( RC_BAD( rc = m_pFileHdl->Read( m_uiFileOffset, m_uiBlockSize,
		m_pucBuffer, &m_uiBlockEnd)))
	{
		if( rc == FERR_IO_END_OF_FILE)
		{
			if( !m_uiBlockEnd)
			{
				goto Exit;
			}
			else
			{
				rc = FERR_OK;
			}
		}
	}

	m_uiFileOffset += m_uiBlockEnd;
	m_pucBufPos = m_pucBuffer;

Exit:

	return( rc);
}
