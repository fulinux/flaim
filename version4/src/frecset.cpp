//-------------------------------------------------------------------------
// Desc:	Record set class implementation
// Tabs:	3
//
//		Copyright (c) 1999-2000,2003,2005-2006 Novell, Inc. All Rights Reserved.
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
// $Id: frecset.cpp 12282 2006-01-19 14:52:56 -0700 (Thu, 19 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#include "flaimsys.h"

/****************************************************************************
Desc: 	Destructor
****************************************************************************/
FlmRecordSet::~FlmRecordSet()
{
	clear();
	if( m_ppRecArray)
	{
		f_free( &m_ppRecArray);
	}
}

/****************************************************************************
Public: 	FlmRecordSet::insert
Desc: 	Insert a FlmRecord into the set.
****************************************************************************/
RCODE FlmRecordSet::insert(
	FlmRecord *		pRecord)
{
	RCODE				rc = FERR_OK;
	FlmRecord **	ppTmpArray;

	// See if we need to reallocate the array.

	if (m_iTotalRecs == m_iRecArraySize)
	{
		if( RC_BAD( rc = f_calloc( 
						sizeof( FlmRecord *) * (m_iRecArraySize + 10),
						&ppTmpArray)))
		{
			goto Exit;
		}
		if (m_iTotalRecs)
		{
			f_memcpy( ppTmpArray, m_ppRecArray,
						sizeof( FlmRecord *) * m_iTotalRecs);
		}
		m_ppRecArray = ppTmpArray;
		m_iRecArraySize += 10;
	}

	// Add the new entry into the array.

	m_ppRecArray [m_iTotalRecs] = pRecord;
	pRecord->AddRef();
	m_iTotalRecs++;

Exit:
	return( rc);
}

/****************************************************************************
Public: 	FlmRecordSet::clear
Desc: 	Clear all records from the FlmRecord set.
****************************************************************************/
void FlmRecordSet::clear( void)
{
	FLMINT	iCnt;

	for (iCnt = 0; iCnt < m_iTotalRecs; iCnt++)
	{
		m_ppRecArray [iCnt]->Release();
		m_ppRecArray [iCnt] = NULL;
	}

	// Set is now empty.

	m_iTotalRecs = 0;

	// Reset the current position.

	m_iCurrRec = -1;
}

/****************************************************************************
Public: 	FlmRecordSet::next
Desc: 	Return the next record in the set.
****************************************************************************/
FlmRecord * FlmRecordSet::next( void)
{
	if (!m_iTotalRecs)
	{
		return( NULL);
	}
	if (m_iCurrRec + 1 >= m_iTotalRecs)
	{
		m_iCurrRec = m_iTotalRecs;
		return( NULL);
	}
	m_iCurrRec++;
	return( m_ppRecArray [m_iCurrRec]);
}
