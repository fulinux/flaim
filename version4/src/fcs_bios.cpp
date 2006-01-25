//-------------------------------------------------------------------------
// Desc:	Class for a buffer stream for I/O operations.
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
// $Id: fcs_bios.cpp 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#include "flaimsys.h"

/****************************************************************************
Desc:
*****************************************************************************/
FCS_BIOS::FCS_BIOS( void)
{
	GedPoolInit( &m_pool, (FCS_BIOS_BLOCK_SIZE + sizeof( FCSBIOSBLOCK)) * 2);
	m_bMessageActive = FALSE;
	m_pRootBlock = NULL;
	m_pCurrWriteBlock = NULL;
	m_pCurrReadBlock = NULL;
	m_bAcceptingData = FALSE;
	m_pEventHook = NULL;
	m_pvUserData = 0;
}

/****************************************************************************
Desc:
*****************************************************************************/
FCS_BIOS::~FCS_BIOS()
{
	GedPoolFree( &m_pool);
}

/****************************************************************************
Desc:	Clears all pending data
*****************************************************************************/
RCODE FCS_BIOS::reset( void)
{
	return( close());
}

/****************************************************************************
Desc:	Flushes any pending data and closes the stream.
*****************************************************************************/
RCODE FCS_BIOS::close( void)
{
	RCODE		rc = FERR_OK;

	GedPoolReset( &m_pool, NULL);
	m_bMessageActive = FALSE;
	m_pRootBlock = NULL;
	m_pCurrWriteBlock = NULL;
	m_pCurrReadBlock = NULL;
	m_bAcceptingData = FALSE;

	return( rc);
}


/****************************************************************************
Desc:	Writes the requested amount of data to the stream.
*****************************************************************************/
RCODE FCS_BIOS::write(
	FLMBYTE *		pucData,
	FLMUINT			uiLength)
{
	FLMUINT				uiCopySize;
	FLMUINT				uiDataPos = 0;
	FCSBIOSBLOCK *		pPrevBlock = NULL;
	RCODE					rc = FERR_OK;

	if( !m_bAcceptingData)
	{
		GedPoolReset( &m_pool, NULL);
		m_pCurrWriteBlock = NULL;
		m_pCurrReadBlock = NULL;
		m_pRootBlock = NULL;
		m_bAcceptingData = TRUE;
	}

	while( uiLength)
	{
		if( !m_pCurrWriteBlock ||
			m_pCurrWriteBlock->uiCurrWriteOffset == FCS_BIOS_BLOCK_SIZE)
		{
			pPrevBlock = m_pCurrWriteBlock;
			m_pCurrWriteBlock =
				(FCSBIOSBLOCK *)GedPoolCalloc( &m_pool, sizeof( FCSBIOSBLOCK));
			if( !m_pCurrWriteBlock)
			{
				rc = RC_SET( FERR_MEM);
				goto Exit;
			}

			m_pCurrWriteBlock->pucBlock =
				(FLMBYTE *)GedPoolAlloc( &m_pool, FCS_BIOS_BLOCK_SIZE);

			if( !m_pCurrWriteBlock->pucBlock)
			{
				rc = RC_SET( FERR_MEM);
				goto Exit;
			}

			if( pPrevBlock)
			{
				pPrevBlock->pNextBlock = m_pCurrWriteBlock;
			}
			else
			{
				m_pRootBlock = m_pCurrWriteBlock;
				m_pCurrReadBlock = m_pCurrWriteBlock;
			}
		}

		uiCopySize = f_min( uiLength,
			(FLMUINT)(FCS_BIOS_BLOCK_SIZE -
			m_pCurrWriteBlock->uiCurrWriteOffset));

		flmAssert( uiCopySize != 0);

		f_memcpy( &(m_pCurrWriteBlock->pucBlock[
			m_pCurrWriteBlock->uiCurrWriteOffset]),
			&(pucData[ uiDataPos]), uiCopySize);
		
		m_pCurrWriteBlock->uiCurrWriteOffset += uiCopySize;
		uiDataPos += uiCopySize;
		uiLength -= uiCopySize;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Terminates the current message
*****************************************************************************/
RCODE FCS_BIOS::endMessage( void)
{
	RCODE		rc = FERR_OK;

	if( !m_bAcceptingData)
	{
		goto Exit;
	}

	if( m_pEventHook)
	{
		if( RC_BAD( rc = m_pEventHook( this,
			FCS_BIOS_EOM_EVENT, m_pvUserData)))
		{
			goto Exit;
		}
	}

Exit:

	m_bAcceptingData = FALSE;
	return( rc);
}


/****************************************************************************
Desc:	Reads the requested amount of data from the stream.
*****************************************************************************/
RCODE FCS_BIOS::read(
	FLMBYTE *		pucData,
	FLMUINT			uiLength,
	FLMUINT *		puiBytesRead)
{
	FLMUINT		uiCopySize;
	FLMUINT		uiDataPos = 0;
	RCODE			rc = FERR_OK;

	if( puiBytesRead)
	{
		*puiBytesRead = 0;
	}

	if( m_bAcceptingData)
	{
		m_bAcceptingData = FALSE;
	}
	
	while( uiLength)
	{
		if( m_pCurrReadBlock &&
			m_pCurrReadBlock->uiCurrReadOffset == m_pCurrReadBlock->uiCurrWriteOffset)
		{
			m_pCurrReadBlock = m_pCurrReadBlock->pNextBlock;
		}

		if( !m_pCurrReadBlock)
		{
			GedPoolReset( &m_pool, NULL);
			rc = RC_SET( FERR_EOF_HIT);
			goto Exit;
		}

		uiCopySize = f_min( uiLength,
			m_pCurrReadBlock->uiCurrWriteOffset - m_pCurrReadBlock->uiCurrReadOffset);

		f_memcpy( &(pucData[ uiDataPos]),
			&(m_pCurrReadBlock->pucBlock[ m_pCurrReadBlock->uiCurrReadOffset]),
			uiCopySize);

		m_pCurrReadBlock->uiCurrReadOffset += uiCopySize;
		uiDataPos += uiCopySize;

		if( puiBytesRead)
		{
			(*puiBytesRead) += uiCopySize;
		}
		uiLength -= uiCopySize;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	
*****************************************************************************/
FLMBOOL FCS_BIOS::isDataAvailable( void)
{
	if( m_bAcceptingData)
	{
		if( m_pRootBlock && m_pRootBlock->uiCurrWriteOffset)
		{
			return( TRUE);
		}
	}
	else if( m_pCurrReadBlock &&
		((m_pCurrReadBlock->uiCurrReadOffset <
			m_pCurrReadBlock->uiCurrWriteOffset) ||
		(m_pCurrReadBlock->pNextBlock)))
	{
		return( TRUE);
	}

	return( FALSE);
}


/****************************************************************************
Desc:	Returns the amount of data available for reading
*****************************************************************************/
FLMUINT FCS_BIOS::getAvailable( void)
{
	FLMUINT				uiAvail = 0;
	FCSBIOSBlock *		pBlk;

	if( m_bAcceptingData)
	{
		if( m_pRootBlock && m_pRootBlock->uiCurrWriteOffset)
		{
			pBlk = m_pRootBlock;
			while( pBlk)
			{
				uiAvail += pBlk->uiCurrWriteOffset;
				pBlk = pBlk->pNextBlock;
			}
		}
	}
	else if( m_pCurrReadBlock &&
		((m_pCurrReadBlock->uiCurrReadOffset <
			m_pCurrReadBlock->uiCurrWriteOffset) ||
		(m_pCurrReadBlock->pNextBlock)))
	{
		pBlk = m_pCurrReadBlock;
		while( pBlk)
		{
			uiAvail += (pBlk->uiCurrWriteOffset -
				pBlk->uiCurrReadOffset);
			pBlk = pBlk->pNextBlock;
		}
	}

	return( uiAvail);
}
