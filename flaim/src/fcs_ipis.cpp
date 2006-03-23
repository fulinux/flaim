//-------------------------------------------------------------------------
// Desc:	TCP/IP input stream class.
// Tabs:	3
//
//		Copyright (c) 1998-2000,2003,2005-2006 Novell, Inc. All Rights Reserved.
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
// $Id: fcs_ipis.cpp 12251 2006-01-19 14:33:30 -0700 (Thu, 19 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#include "flaimsys.h"

/****************************************************************************
Desc:	Constructor
*****************************************************************************/
FCS_IPIS::FCS_IPIS( FCS_TCP * pTcpObj)
{
	m_pTcpObj = pTcpObj;
	m_pucBufPos = m_pucBuffer;
	m_bStreamInvalid = FALSE;
	m_bMessageActive = FALSE;
	m_bGotLastPacket = FALSE;
	m_uiPacketSize = 0;
}

/****************************************************************************
Desc:
*****************************************************************************/
FCS_IPIS::~FCS_IPIS( void)
{
	(void)close();
}

/****************************************************************************
Desc:
*****************************************************************************/
FLMBOOL FCS_IPIS::isOpen( void)
{
	return( TRUE);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE FCS_IPIS::close( void)
{
	(void)endMessage();
	m_bStreamInvalid = FALSE;
	return( FERR_OK);
}

/****************************************************************************
Desc:	Reads the requested amount of data from the stream.
*****************************************************************************/
RCODE FCS_IPIS::read(
	FLMBYTE *		pucData,
	FLMUINT			uiLength,
	FLMUINT *		puiBytesRead)
{
	FLMUINT	uiBytesRead = 0;
	FLMUINT	uiMaxSize;
	RCODE		rc = FERR_OK;

	if( puiBytesRead)
	{
		*puiBytesRead = 0;
	}

	if( !m_bStreamInvalid)
	{
		while( uiLength)
		{
			uiMaxSize = m_uiPacketSize - (FLMUINT)(m_pucBufPos - m_pucBuffer);

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
	}
	else
	{
		rc = RC_SET( FERR_READING_FILE);
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
RCODE FCS_IPIS::flush( void)
{
	RCODE		rc = FERR_OK;

	if( !m_bMessageActive)
	{
		goto Exit;
	}

	for( ;;)
	{
		if( RC_BAD( rc = getNextPacket()))
		{
			if( rc == FERR_EOF_HIT)
			{
				rc = FERR_OK;
			}
			goto Exit;
		}
	}

Exit:

	m_pucBufPos = m_pucBuffer;
	return( rc);
}


/****************************************************************************
Desc:	Flushes any pending data.
*****************************************************************************/
RCODE FCS_IPIS::endMessage( void)
{
	RCODE		rc = FERR_OK;

	if( !m_bMessageActive)
	{
		goto Exit;
	}

	if( RC_BAD( rc = flush()))
	{
		goto Exit;
	}

Exit:

	m_bMessageActive = FALSE;
	m_bGotLastPacket = FALSE;
	return( rc);
}


/****************************************************************************
Desc:	Reads the next packet off the wire.
*****************************************************************************/
RCODE FCS_IPIS::getNextPacket( void)
{
	FLMBYTE		pucDescriptor[ 2];
	FLMUINT		uiDescriptor;
	FLMUINT		uiActualCnt = 0;
	RCODE			rc = FERR_OK;

	if( !m_bStreamInvalid)
	{
		if( !m_bMessageActive)
		{
			m_bMessageActive = TRUE;
		}

		if( m_bGotLastPacket)
		{
			rc = RC_SET( FERR_EOF_HIT);
			goto Exit;
		}
	
		if( RC_BAD( rc = m_pTcpObj->readAll( pucDescriptor,
			2, &uiActualCnt)))
		{
			goto Exit;
		}
		
		uiDescriptor = flmBigEndianToUINT16( pucDescriptor);
		m_uiPacketSize = uiDescriptor & 0x7FFF;

		if( uiDescriptor & 0x8000)
		{
			m_bGotLastPacket = TRUE;
		}

		if( m_uiPacketSize > FCS_IPIS_BUFFER_SIZE)
		{
			m_uiPacketSize = 0;
			rc = RC_SET( FERR_READING_FILE);
			goto Exit;
		}

		if( m_uiPacketSize > 0)
		{
			if( RC_BAD( rc = m_pTcpObj->readAll( m_pucBuffer,
				m_uiPacketSize, &uiActualCnt)))
			{
				goto Exit;
			}
		}
		else
		{
			if( m_bGotLastPacket)
			{
				rc = RC_SET( FERR_EOF_HIT);
			}
			else
			{
				rc = RC_SET( FERR_READING_FILE);
			}
			goto Exit;
		}
	
		m_pucBufPos = m_pucBuffer;
	}
	else
	{
		rc = RC_SET( FERR_READING_FILE);
	}

Exit:

	if( RC_BAD( rc) && rc != FERR_EOF_HIT)
	{
		m_bStreamInvalid = TRUE;
	}

	return( rc);
}
