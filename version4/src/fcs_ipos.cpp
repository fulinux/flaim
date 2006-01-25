//-------------------------------------------------------------------------
// Desc:	TCP/IP output stream class.
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
// $Id: fcs_ipos.cpp 12251 2006-01-19 14:33:30 -0700 (Thu, 19 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#include "flaimsys.h"

/****************************************************************************
Desc:
*****************************************************************************/
FCS_IPOS::FCS_IPOS( FCS_TCP * pTcpObj)
{
	m_pTcpObj = pTcpObj;
	m_pucBufPos = &(m_pucBuffer[ 2]);
	m_bOpen = TRUE;
	m_bMessageActive = FALSE;
}


/****************************************************************************
Desc:	Flushes any pending data and closes the stream.
*****************************************************************************/
RCODE FCS_IPOS::close( void)
{
	RCODE		rc = FERR_OK;

	if( m_bOpen)
	{
		rc = endMessage();
		m_bOpen = FALSE;
	}

	return( rc);
}


/****************************************************************************
Desc:	Writes the requested amount of data to the stream.
*****************************************************************************/
RCODE FCS_IPOS::write(
	FLMBYTE *		pucData,
	FLMUINT			uiLength)
{
	FLMUINT	uiMaxSize;
	RCODE		rc = FERR_OK;

	if( !uiLength)
	{
		goto Exit;
	}

	if( m_bOpen)
	{
		while( uiLength)
		{
			uiMaxSize =
				(FLMUINT)(FCS_IPOS_BUFFER_SIZE - (m_pucBufPos - m_pucBuffer));

			if( !uiMaxSize)
			{
				if( RC_BAD( rc = flush()))
				{
					goto Exit;
				}
			}
			else if( uiLength <= uiMaxSize)
			{
				f_memcpy( m_pucBufPos, pucData, uiLength);
				m_pucBufPos += uiLength;
				uiLength = 0;
			}
			else
			{
				f_memcpy( m_pucBufPos, pucData, uiMaxSize);
				m_pucBufPos += uiMaxSize;
				pucData += uiMaxSize;
				uiLength -= uiMaxSize;
				if( RC_BAD( rc = flush()))
				{
					goto Exit;
				}
			}
		}
		m_bMessageActive = TRUE;
	}
	else
	{
		rc = RC_SET( FERR_WRITING_FILE);
	}

Exit:

	return( rc);
}


/****************************************************************************
Desc:	Flushes any pending data and optionally ends the current message.
*****************************************************************************/
RCODE	FCS_IPOS::_flush(
	FLMBOOL		bEndMessage)
{
	FLMUINT		uiActualCnt;
	FLMUINT		uiLength;
	FLMUINT		uiDescriptor;
	RCODE			rc = FERR_OK;

	if( (uiLength = (FLMUINT)(m_pucBufPos - m_pucBuffer)) != 0)
	{
		uiDescriptor = uiLength - 2;
		if( bEndMessage)
		{
			uiDescriptor |= 0x8000;
		}

		if( uiDescriptor)
		{
			intToByte( (FLMUINT16)uiDescriptor, m_pucBuffer);

			if( RC_BAD( rc = m_pTcpObj->write( m_pucBuffer,
				uiLength, &uiActualCnt)))
			{
				goto Exit;
			}
		}
	}

Exit:

	m_pucBufPos = &(m_pucBuffer[ 2]);
	return( rc);
}


/****************************************************************************
Desc:	Terminates the current message
*****************************************************************************/
RCODE FCS_IPOS::endMessage( void)
{
	RCODE		rc = FERR_OK;


	if( !m_bMessageActive)
	{
		goto Exit;
	}

	if( RC_BAD( rc = _flush( TRUE)))
	{
		goto Exit;
	}

Exit:

	m_bMessageActive = FALSE;
	return( rc);
}
