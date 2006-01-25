//------------------------------------------------------------------------------
// Desc:	This file contains the routines to initialize and set up
//			structures for indexing.
//
// Tabs:	3
//
//		Copyright (c) 1992-2000, 2002-2006 Novell, Inc. All Rights Reserved.
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
// $Id: kyunlock.cpp 3115 2006-01-19 13:24:39 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------

#include "flaimsys.h"

/****************************************************************************
Desc:	Setup routine for the KREF_CNTRL structure for record updates.
****************************************************************************/
RCODE F_Db::krefCntrlCheck( void)
{
	RCODE	rc = NE_XFLM_OK;

	// Check if we need to flush keys between updates, but not during the
	// processing of an update.

	if( m_bKrefSetup)
	{
		if( isKrefOverThreshold() ||
			 (m_pOldNodeList && m_pOldNodeList->getNodeCount()))
		{
			if (RC_BAD( rc = keysCommit( FALSE)))
			{
				goto Exit;
			}
		}
	}
	else
	{
		m_uiKrefCount = 0;
		m_uiTotalKrefBytes = 0;
		m_pKrefPool = NULL;
		m_bReuseKrefPool = FALSE;
		m_bKrefCompoundKey = FALSE;
		m_pKrefReset = NULL;
		m_bKrefSetup = TRUE;

		if (m_eTransType == XFLM_UPDATE_TRANS)
		{
			m_pKrefPool = &m_pDatabase->m_krefPool;
			m_bReuseKrefPool = TRUE;
			m_pKrefPool->poolReset( NULL, TRUE);
		}
		else
		{
			m_pKrefPool = &m_tmpKrefPool;
			m_bReuseKrefPool = FALSE;
			m_pKrefPool->poolInit( DEFAULT_KREF_POOL_BLOCK_SIZE);
		}

		if( !m_pKrefTbl)
		{
			if( RC_BAD( rc = f_alloc( 
				DEFAULT_KREF_TBL_SIZE * sizeof( KREF_ENTRY *), &m_pKrefTbl)))
			{
				goto Exit;
			}

			m_uiKrefTblSize = DEFAULT_KREF_TBL_SIZE;
		}
		
		if( !m_pucKrefKeyBuf)
		{
			if (RC_BAD( rc = f_alloc( MAX_KEY_SIZ, &m_pucKrefKeyBuf)))
			{
				goto Exit;
			}
		}
	}

	m_pKrefReset = m_pKrefPool->poolMark();
	flmAssert( m_pucKrefKeyBuf);

Exit:

	if (RC_BAD( rc))
	{
		krefCntrlFree();
	}

	return( rc);
}

/****************************************************************************
Desc:	Frees the memory associated with the KREF
****************************************************************************/
void F_Db::krefCntrlFree( void)
{
	if( m_bKrefSetup)
	{
		if( m_bReuseKrefPool)
		{
			m_pKrefPool->poolReset( NULL, TRUE);
		}
		else
		{
			m_pKrefPool->poolFree();
		}

		if( m_pKrefTbl && m_uiKrefTblSize != DEFAULT_KREF_TBL_SIZE)
		{
			f_free( &m_pKrefTbl);
			m_uiKrefTblSize = 0;
		}

		m_uiKrefCount = 0;
		m_uiTotalKrefBytes = 0;
		m_pKrefPool = NULL;
		m_bReuseKrefPool = FALSE;
		m_bKrefCompoundKey = FALSE;
		m_pKrefReset = NULL;
		m_bKrefSetup = FALSE;
		
		if (m_pOldNodeList)
		{
			m_pOldNodeList->resetList();
		}
	}
}
