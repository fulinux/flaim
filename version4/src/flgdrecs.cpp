//-------------------------------------------------------------------------
// Desc:	Routines for adding, modifying, and deleting fields in a FlmRecord.
// Tabs:	3
//
//		Copyright (c) 1995-2001,2003,2005-2006 Novell, Inc. All Rights Reserved.
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
//		PERPETRATOR TO CRIMINAL AND CIVIL LIABILITY.
//
// $Id: flgdrecs.cpp 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#include "flaimsys.h"

#define UBUF_SIZE 32

/****************************************************************************
Desc: 	This routine adds a field to a record.
****************************************************************************/
RCODE flmAddField(
	FlmRecord *		pRecord,
	FLMUINT			uiTagNum,
	const void *	pvData,
	FLMUINT			uiDataLen,
	FLMUINT			uiDataType)
{
	RCODE				rc = FERR_OK;
	void *			pvField;

	// Insert new field. 
	
	if( RC_BAD( rc = pRecord->insertLast( 1, uiTagNum, uiDataType, &pvField)))
	{
		goto Exit;
	}

	switch( uiDataType)
	{
		case FLM_TEXT_TYPE:
		{
			rc = pRecord->setNative( pvField, (const char *)pvData);

			break;
		}

		case FLM_NUMBER_TYPE:
		{
			FLMUINT	uiNum;

			switch (uiDataLen)
			{
				case 0:
					uiNum = (FLMUINT)(*((FLMUINT *)(pvData)));
					break;
				case 1:
					uiNum = (FLMUINT)(*((FLMBYTE *)(pvData)));
					break;
				case 2:
					uiNum = (FLMUINT)(*((FLMUINT16 *)(pvData)));
					break;
				case 4:
					uiNum = (FLMUINT)(*((FLMUINT32 *)(pvData)));
					break;
				default:
					flmAssert( 0);
					rc = RC_SET( FERR_INVALID_PARM);
					goto Exit;
			}
			rc = pRecord->setUINT( pvField, uiNum);
			break;
		}
		case FLM_BINARY_TYPE:
		{
			rc = pRecord->setBinary( pvField, pvData, uiDataLen);
			break;
		}
		default :
		{
			flmAssert( 0);
			break;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc: 	This routine modifies the first matching field in a record.
			If the field is not found, a new field will be created.
****************************************************************************/
RCODE flmModField(
	FlmRecord *		pRecord,
	FLMUINT			uiTagNum,
	const void *	pvData,
	FLMUINT			uiDataLen,
	FLMUINT			uiDataType)
{
	RCODE				rc = FERR_OK;
	void *			pvField;

	if( (pvField = pRecord->find( pRecord->root(), uiTagNum)) == NULL)
	{
		// Create the field.
		
		if( RC_BAD( rc = pRecord->insertLast( 1, uiTagNum, uiDataType, &pvField)))
		{
			goto Exit;
		}
	}
	
	switch( uiDataType)
	{
		case FLM_TEXT_TYPE:
		{
			rc = pRecord->setNative( pvField, (const char *)pvData);
			break;
		}
		
		case FLM_NUMBER_TYPE:
		{
			FLMUINT	uiNum;
			switch (uiDataLen)
			{
				case 0:
					uiNum = (FLMUINT)(*((FLMUINT *)(pvData)));
				case 1:
					uiNum = (FLMUINT)(*((FLMBYTE *)(pvData)));
					break;
				case 2:
					uiNum = (FLMUINT)(*((FLMUINT16 *)(pvData)));
					break;
				case 4:
					uiNum = (FLMUINT)(*((FLMUINT32 *)(pvData)));
					break;
				default:
					flmAssert( 0);
					rc = RC_SET( FERR_INVALID_PARM);
					goto Exit;
			}
			
			rc = pRecord->setUINT( pvField, uiNum);
			break;
		}
		
		case FLM_BINARY_TYPE:
		{
			rc = pRecord->setBinary( pvField, pvData, uiDataLen);
			break;
		}
		
		default :
		{
			flmAssert( 0);
			break;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc: 	This routine searches for a specific numeric field and deletes
			that field from the record.
****************************************************************************/
RCODE flmDelField(
	FlmRecord *	pRecord,
	FLMUINT		uiTagNum,
	FLMUINT		uiValue)
{
	RCODE			rc = FERR_OK;
	FLMUINT		uiNum;
	void *		pvField;

	if( (pvField = pRecord->find( pRecord->root(), uiTagNum, 1)) != NULL)
	{
		for(;;)
		{
			if( pRecord->getFieldID( pvField) == uiTagNum)
			{
				if( RC_BAD( rc = pRecord->getUINT( pvField, &uiNum)))
				{
					goto Exit;
				}

				if( uiNum == uiValue)
				{
					pRecord->remove( pvField);
					break;
				}
			}
			
			pvField = pRecord->nextSibling( pvField);
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc: 	This routine finds a field in a record and increments its value.
			The value of 1 will be assigned if the field is not present.
****************************************************************************/
RCODE flmIncrField(
	FlmRecord *		pRecord,		
	FLMUINT			uiTagNum)
{
	RCODE				rc = FERR_OK;
	void *			pvField;

	if( (pvField = pRecord->find( pRecord->root(), uiTagNum, 1)) != NULL)
	{
		FLMUINT		uiNum;

		if( RC_OK( rc = pRecord->getUINT( pvField, &uiNum)))
		{
			uiNum++;
			rc = pRecord->setUINT( pvField, uiNum);
		}
	}
	else
	{
		// Create the field and set the value to one.
		
		if( RC_OK( rc = pRecord->insertLast( 1, uiTagNum, 
			FLM_NUMBER_TYPE, &pvField)))
		{
			rc = pRecord->setUINT( pvField, 1);
		}
	}

	return( rc);
}

/****************************************************************************
Desc: 	This routine finds a field in a record and decrements its value.
****************************************************************************/
RCODE flmDecrField(
	FlmRecord *		pRecord,
	FLMUINT			uiTagNum)
{
	RCODE				rc = FERR_OK;
	void *			pvField;

	if( (pvField = pRecord->find( pRecord->root(), uiTagNum, 1)) != NULL)
	{
		FLMUINT		uiNum;

		if( RC_OK( rc = pRecord->getUINT( pvField, &uiNum)))
		{
			uiNum--;
			rc = pRecord->setUINT( pvField, uiNum);
		}
	}
	
	return( rc);
}

/****************************************************************************
Desc:		This routine adds a field to a GEDCOM tree
****************************************************************************/
RCODE gedAddField(
	POOL *			pPool,
	NODE *			pRecord,
	FLMUINT			uiTagNum,
	const void *	pvData,
	FLMUINT			uiDataLen,
	FLMUINT			uiDataType)
{
	RCODE			rc = FERR_OK;
	NODE *		pChildNode;
	FLMUINT		uiNum;

	if ((pChildNode = GedNodeMake( pPool, uiTagNum, &rc)) == NULL)
	{
		goto Exit;
	}

	switch( uiDataType)
	{
		case FLM_TEXT_TYPE:
		{
			rc = GedPutNATIVE( pPool, pChildNode, (const char *)pvData);
			break;
		}
		
		case FLM_NUMBER_TYPE:
		{
			switch (uiDataLen)
			{
				case 0:
					uiNum = (FLMUINT)(*((FLMUINT *)(pvData)));
					break;
				case 1:
					uiNum = (FLMUINT)(*((FLMBYTE *)(pvData)));
					break;
				case 2:
					uiNum = (FLMUINT)(*((FLMUINT16 *)(pvData)));
					break;
				case 4:
					uiNum = (FLMUINT)(*((FLMUINT32 *)(pvData)));
					break;
				default:
					flmAssert( 0);
					rc = RC_SET( FERR_INVALID_PARM);
					goto Exit;
			}
			
			rc = GedPutUINT( pPool, pChildNode, uiNum);
			break;
		}
		
		case FLM_BINARY_TYPE:
		{
			rc = GedPutBINARY( pPool, pChildNode, pvData, uiDataLen);
			break;
		}
	}
	
	if (RC_BAD( rc))
	{
		goto Exit;
	}
	
	GedChildGraft( pRecord, pChildNode, GED_LAST);

Exit:

	return( rc);
}
