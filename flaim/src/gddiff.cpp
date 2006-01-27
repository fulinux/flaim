//-------------------------------------------------------------------------
// Desc:	Difference two FlmRecord objects.
// Tabs:	3
//
//		Copyright (c) 1998-2000,2003-2006 Novell, Inc. All Rights Reserved.
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
// $Id: gddiff.cpp 12304 2006-01-19 15:04:25 -0700 (Thu, 19 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#include "flaimsys.h"

class RecCursor : public F_Base
{
public:

	/* Instance methods ------------------------------------------------------*/

	RecCursor(								// Constructor
		FlmRecord *		pRecord,
		GRD_CallBackFunction	pCallBackFunction,
		void *			pvCallBackData)
	{
		m_pRecord = pRecord;
		m_pvField = pRecord ? pRecord->root() : NULL;
		m_uiRootLevel = pRecord ? pRecord->getLevel( m_pvField) : 0 ;
		m_uiAbsoluteCursorPosition = 1;
		m_pCallBack = pCallBackFunction;
		m_pvCallBackData = pvCallBackData;
		m_bStillAtTheRoot = TRUE;
	}

	RecCursor(								// Copy Constructor
		RecCursor *		cursor)
	{
		m_pRecord = cursor->m_pRecord;
		m_pvField = cursor->m_pvField;
		m_uiRootLevel = cursor->m_uiRootLevel;
		m_uiAbsoluteCursorPosition = cursor->m_uiAbsoluteCursorPosition;
		m_pCallBack = cursor->m_pCallBack;
		m_pvCallBackData = cursor->m_pvCallBackData;
		m_bStillAtTheRoot = cursor->m_bStillAtTheRoot;
	}
		
	virtual ~RecCursor()					// Destructor
	{
		if (m_pRecord)
		{
			m_pRecord = NULL;
		}
	}

	FINLINE FLMBOOL EndOfRecord()		// TRUE = End-of-Record has been reached
	{
		return( 
			m_pvField == NULL ? TRUE 
				: (m_pRecord->getLevel( m_pvField) <= m_uiRootLevel
					&& !m_bStillAtTheRoot) ? TRUE : FALSE);
	}
	
	FINLINE void Advance()				// Advance cursor to next field
	{
		m_bStillAtTheRoot = FALSE;
		if ( m_pvField)
		{
			m_pvField = m_pRecord->next( m_pvField);
			m_uiAbsoluteCursorPosition++;
		}
	}

	FLMBOOL FieldValueIsEqualTo(
		RecCursor	* pSomeField);


	/****************************************************************************
	Desc:		Determine if the IDs of 2 fields are equal
	Return:	TRUE = the IDs are equal
	ASSUMPTIONS: pField1 and pField2 are not NULL
	****************************************************************************/
	FINLINE FLMBOOL FieldIdIsEqualTo(
		RecCursor	* pSomeField)
	{
		return(
			Level() == pSomeField->Level()
			&& m_pRecord->getFieldID( m_pvField) == 
				pSomeField->m_pRecord->getFieldID( pSomeField->m_pvField)
			&& m_pRecord->getDataType( m_pvField) == 
				pSomeField->m_pRecord->getDataType( pSomeField->m_pvField)
			? TRUE : FALSE);
	}

	
	enum RecFieldMatchTypes
	{
		GRD_NoMatch,	// no match was found
		GRD_ExactMatch, // an exact match was found
		GRD_IDMatch		// A field with the same ID was found, but the 
							//	value was different.  ID = level+tNum+type
	};

	void * Scan(
		RecCursor	* pTargetCursor,
		RecFieldMatchTypes *	peMatchType);


	FINLINE FLMUINT AbsolutePosition()// Return the 1-based position of the 
	{											// current field
		return( m_uiAbsoluteCursorPosition );
	}

	FINLINE void * Field()				// Return a pointer to the current field
	{
		return( m_pvField);
	}
	
	FINLINE FlmRecord * Record()		// Return a pointer to the current record
	{
		return( m_pRecord);
	}

	FINLINE FLMUINT Level()				// Return the normalized level of the 
	{											// current field
		return( m_pvField ? Normalize( m_pRecord->getLevel( m_pvField)) : 0 );
	}
	
	FINLINE FLMUINT RawLevel()			// Return the level in its raw, 
	{											// unnormalized form.
		return( m_pvField ? m_pRecord->getLevel( m_pvField) : 0 );
	}
	
	FINLINE void CallBack( 				// Call the caller's callback function
		GRD_DifferenceData	&difference)
	{
		(*m_pCallBack)( difference, m_pvCallBackData );
	}



	/* Class methods ---------------------------------------------------------*/
	
	static void MarkBranchDeleted(
		RecCursor	*	pBeforeCursor,
		RecCursor	*	pAfterCursor);

	static void MarkModified(
		RecCursor	*	pBeforeCursor,
		RecCursor	*	pAfterCursor);

	static void MarkInserted(
		RecCursor	*	pCursor);
		
	static void MarkRangeInserted(
		RecCursor	*	pAfterCursor,
		void *			pEndOfRange);
		
private:

	FLMUINT				m_uiAbsoluteCursorPosition;// 1-based Absolute position of cursor
	FlmRecord *			m_pRecord;				// Pointer to current node
	void *				m_pvField;
	FLMUINT				m_uiRootLevel;			// Level of the root field
	GRD_CallBackFunction	m_pCallBack;// Pointer to caller's callback function
	void *				m_pvCallBackData;		// Pointer to caller's data
	FLMBOOL				m_bStillAtTheRoot;	// TRUE = cursor is still on the root field
	

	/* Methods  */
	RecCursor(){}							// Not allowed

	FINLINE FLMUINT Normalize(
		FLMUINT			level)
	{	/* Field levels must be normalized when comparing the levels of two fields
			from different records.  This allows the "root fields" of the
			2 records to have different levels. */
		 return( level - m_uiRootLevel );
	}
	
	FINLINE FLMBOOL isLeafField()
	{
		void *	pvNext = m_pRecord->next( m_pvField);
		
		//It is valid to compare raw node levels of nodes within the same record
		return(
			(pvNext
			 && m_pRecord->getLevel( pvNext) > m_pRecord->getLevel( m_pvField) )
			 	? FALSE	// Field has children, not a leaf field.
				: TRUE);
	}
};


/*
 * Large instance method implementations
 */

/****************************************************************************
Desc:		Determine if 2 fields' values are equal
Return:	TRUE = the 2 values are equal
ASSUMPTIONS: pField1 and pField2 are not NULL
****************************************************************************/
FLMBOOL RecCursor::FieldValueIsEqualTo(
	RecCursor *			pSomeField)
{
	FLMBOOL				bEqual = FALSE;
	FLMUINT				uiFieldLen = m_pRecord->getDataLength( m_pvField);
	FLMUINT				uiSomeLen = pSomeField->m_pRecord->getDataLength( pSomeField->m_pvField);
	FLMUINT				uiEncFieldLen  = 0;
	FLMUINT				uiEncSomeLen = 0;
	const FLMBYTE *	pValue1;
	const FLMBYTE *	pValue2;

	// If the data lengths are not equal, we can exit.

	if( uiFieldLen != uiSomeLen)
	{
		goto Exit;
	}

	// If one field is encrypted and the other is not, then we can exit.

	if ((m_pRecord->isEncryptedField( m_pvField) && 
		  !pSomeField->m_pRecord->isEncryptedField( pSomeField->m_pvField)) ||
		  (!m_pRecord->isEncryptedField( m_pvField) && 
		  pSomeField->m_pRecord->isEncryptedField( pSomeField->m_pvField)))
	{
		goto Exit;
	}

	// If the fields are encrypted, are they using the same encryption scheme?

	if (m_pRecord->isEncryptedField( m_pvField))
	{
		if (m_pRecord->getEncryptionID( m_pvField) != 
			pSomeField->m_pRecord->getEncryptionID( pSomeField->m_pvField))
		{
			goto Exit;
		}
	}

	// If the field is not encrypted, and we have a value length
	
	if( uiFieldLen && !m_pRecord->isEncryptedField( m_pvField))
	{
		pValue1 = m_pRecord->getDataPtr( m_pvField);
		pValue2 = pSomeField->m_pRecord->getDataPtr( pSomeField->m_pvField);

		// If the values are not equal, we can exit.
		if( f_memcmp( pValue1, pValue2, uiFieldLen) != 0 )
		{
			goto Exit;
		}
	}

	// Otherwise, if the field is encrypted, we need to check the encrypted value.

	else if (m_pRecord->isEncryptedField( m_pvField))
	{
		uiEncFieldLen = m_pRecord->getEncryptedDataLength( m_pvField);
		uiEncSomeLen = pSomeField->m_pRecord->getEncryptedDataLength(
																pSomeField->m_pvField);

		// If the encrypted lengths are not equal, we can exit.

		if (uiEncFieldLen != uiEncSomeLen)
		{
			goto Exit;
		}
		
		if (uiEncFieldLen)
		{
			pValue1 = m_pRecord->getEncryptionDataPtr( m_pvField);
			pValue2 = pSomeField->m_pRecord->getEncryptionDataPtr( pSomeField->m_pvField);
			
			// If the encrypted values are not equal, we can exit.

			if( f_memcmp( pValue1, pValue2, uiFieldLen) != 0 )
			{
				goto Exit;
			}
		}
	}

	// If we get this far, the fields are identical.

	bEqual = TRUE;
	
Exit:

	return( bEqual);

}


/****************************************************************************
Desc:		Scan for the field referenced by the target cursor
out:		eMatchType:
				GRD_NoMatch = the target field could not be found in this
								record
				GRD_IDMatch = the target field was found in this record;
								however, the value was changed.
				GRD_ExactMatch = the target field was found in this record
Return:	A pointer to the field in this record that matches the target field.
			If no match was found, return NULL.
****************************************************************************/
void * RecCursor::Scan(
	RecCursor *		pTargetCursor,
	RecFieldMatchTypes *	peMatchType)
{
	void *			pvIDMatch = NULL;
	FLMUINT			uiTargetLevel = pTargetCursor->Level();
	FLMBOOL			bAdvanced = FALSE;
	
	*peMatchType = GRD_NoMatch;
	
	for( RecCursor candidate = this ; 
				candidate.Level() >= uiTargetLevel && !candidate.EndOfRecord() ;
					 candidate.Advance(), bAdvanced = TRUE )
	{
		if( pTargetCursor->FieldIdIsEqualTo( &candidate))
		{
			if( pTargetCursor->FieldValueIsEqualTo( &candidate))
			{
				*peMatchType = GRD_ExactMatch;
				return( candidate.Field() );
			}
			else if( *peMatchType == GRD_NoMatch)
			{
				if ( !bAdvanced && isLeafField())
				{
					/* Only allow ID matches on leaf fields, when cursor hasn't 
						advanced */
					*peMatchType = GRD_IDMatch;
					pvIDMatch = candidate.Field();
				}
			}
		}
	}

	return( pvIDMatch);
}	


/*
 * Class Method implementations
 */


/****************************************************************************
Desc:		Mark the 'before field,' and all of its children, as deleted - the
			field doesn't exist in the 'after record'.
			
Note:		It is valid to compare raw field levels of fields within the same
			record.
****************************************************************************/
void RecCursor::MarkBranchDeleted(
	RecCursor	*	pBeforeCursor,
	RecCursor	*	pAfterCursor)
{
	GRD_DifferenceData	difference;
	FLMUINT					uiStartLevel = pBeforeCursor->RawLevel();

	difference.type = GRD_DeletedSubtree;
	difference.uiAbsolutePosition = pAfterCursor->AbsolutePosition();
	difference.pBeforeRecord = pBeforeCursor->Record();
	difference.pvBeforeField = pBeforeCursor->Field();
	difference.pAfterRecord = NULL;
	difference.pvAfterField = NULL;

	pBeforeCursor->CallBack( difference );
	difference.type = GRD_Deleted;
	do
	{
		pBeforeCursor->CallBack( difference );
		pBeforeCursor->Advance();
	} while( !pBeforeCursor->EndOfRecord()
				&& pBeforeCursor->RawLevel() > uiStartLevel);
}	

/****************************************************************************
Desc:		Mark the field as modified - the value of the field in the 'after
			field' is different than the value in the 'before field'
****************************************************************************/
void RecCursor::MarkModified(
	RecCursor *	pBeforeCursor,
	RecCursor *	pAfterCursor)
{
	GRD_DifferenceData	difference;
	
	difference.type = GRD_Modified;
	difference.uiAbsolutePosition = pAfterCursor->AbsolutePosition();
	difference.pBeforeRecord = pBeforeCursor->Record();
	difference.pvBeforeField = pBeforeCursor->Field();
	difference.pAfterRecord = pAfterCursor->Record();
	difference.pvAfterField = pAfterCursor->Field();
	
	pBeforeCursor->CallBack( difference);
}	

/****************************************************************************
Desc:		Mark the field as inserted - the field doesn't exist in the 'before
			record'
****************************************************************************/
void RecCursor::MarkInserted(
	RecCursor	* pAfterCursor)
{
	GRD_DifferenceData	difference;
	
	difference.type = GRD_Inserted;
	difference.uiAbsolutePosition = pAfterCursor->AbsolutePosition();
	difference.pBeforeRecord = NULL;
	difference.pvBeforeField = NULL;
	difference.pAfterRecord = pAfterCursor->Record();
	difference.pvAfterField = pAfterCursor->Field();
	
	pAfterCursor->CallBack( difference );
}	

/****************************************************************************
Desc:		Mark all of the fields as inserted, starting with the current field 
			and ending with, but not including, the field referenced by
			'pEndOfRange'
****************************************************************************/
void RecCursor::MarkRangeInserted(
	RecCursor * pAfterCursor,
	void *		pEndOfRange)
{
	void *		pvField;
	
	for( pvField = pAfterCursor->Field(); 
		pvField != pEndOfRange ; pvField = pAfterCursor->Field() )
	{	
		/* Note that MarkInserted will advance the field pointer */
		RecCursor::MarkInserted( pAfterCursor);
		pAfterCursor->Advance();
	}
}


/****************************************************************************
Desc:		Find the differences between the 2 records.
Notes:	This algorithm is intended to be accurate; however it does not always
			generate the smallest number of differences.  Here is an example
			where the algorithm generates more differences than the optimal:

			e						e
			 \						 \
			  a1					  a1-prime
			   \					   \
				 v11				    v11
				 |					    |
				 huge-subtree	    huge-subtree
				/						/
			  a2					  a2
			   \				      \
				 v21					 v21
				 |						 |
				 v22					 v22

			In this case, the optimal results would be:
				'a1' was modified
			Unfortunately, this algorithm will generate:
				'a1' was deleted
				'v11' was deleted
				'huge-subtree' was deleted		<-- this is the bad news
				'a1-prime' was inserted
				'v11' was inserted
				'huge-subtree' was inserted	<-- this is the other bad news


			It will take more memory and code to handle cases like these 
			more optimally.
****************************************************************************/

void flmRecordDifference(
	FlmRecord * 		pBefore,		// 'before' record
	FlmRecord * 		pAfter,		// 'after' record
	GRD_CallBackFunction	pCallBackFunction,// call this function for each difference
	void * 				pvCallBackData) // Pass this data as a parameter to the callback
{
	RecCursor			beforeCursor( pBefore, pCallBackFunction, pvCallBackData);
	RecCursor			afterCursor( pAfter, pCallBackFunction, pvCallBackData);
	
	// Iterate through all of the fields in the 'before record'
	while( ! beforeCursor.EndOfRecord())
	{
		void *	pvFound;
		RecCursor::RecFieldMatchTypes eMatchType;
		
		if ( afterCursor.EndOfRecord() )
		{	/* The end of the 'after record' has been reached.  This means that 
				the 'before field' must have been deleted from the 'after record' */
			RecCursor::MarkBranchDeleted( &beforeCursor, &afterCursor );
			continue;
		}

		pvFound = afterCursor.Scan( &beforeCursor, &eMatchType);
		if( pvFound)
		{	
			// 'before field' found in 'after record'

			//Mark all intervening 'after fields' as inserted
			RecCursor::MarkRangeInserted( &afterCursor, pvFound);
			if( eMatchType == RecCursor::GRD_IDMatch)
			{	
				// 'before field' was modified in 'after record'
				RecCursor::MarkModified( &beforeCursor, &afterCursor);
			}
			else /* eMatchType == GRD_ExactMatch */
			{	// 'before field' == 'after field', advance to next field
			}
			afterCursor.Advance();
			beforeCursor.Advance();
		}
		else
		{	
			// 'before field' has been deleted from 'after record'
			RecCursor::MarkBranchDeleted( &beforeCursor, &afterCursor);
		}

	}	/* End of While */

	/* The end of the 'before record' has been reached, all remaining 
		'after fields' must have been inserted */
	RecCursor::MarkRangeInserted( &afterCursor, NULL);
}	


/*
Text differencing
*/

class StringCursor : public F_Base
{

public:
	StringCursor(							//Constructor
		FLMBYTE *		string,
		FLMUINT			length,
		GSD_CallBackFunction	pCallBackFunction,
		void *			pvCallBackData)
	{
		m_pString = string;
		m_uiLength = length;
		m_uiCursor = 0;
		m_pCallBack = pCallBackFunction;
		m_pvCallBackData = pvCallBackData;

		m_eSubState = GSD_Initial;
		m_uiSubStart = 0;
		m_uiSubLength = 0;
	}
	
	virtual ~StringCursor(){}			// Destructor


	FINLINE FLMBOOL EndOfString()			//TRUE = the end of the string has been reached
	{
		return(
			m_pString == NULL ? 
				TRUE
				: m_uiCursor >= m_uiLength ? TRUE : FALSE);
	}
	
	FINLINE void Advance()				//Advance the cursor to the next byte
	{
		if ( m_uiCursor < m_uiLength )
		{
			m_uiCursor++;
		}
	}
	
	FINLINE void Advance(					//Advance the cursor to the given position
		FLMUINT			uiPosition)
	{
		if ( uiPosition < m_uiLength)
		{
			m_uiCursor = uiPosition;
		}
	}
	
	FINLINE void AdvanceToEndOfString()//Advance the cursor to the end of the string
	{
		m_uiCursor = m_uiLength;
	}

	FINLINE FLMBYTE Byte()					//Return the byte the cursor is on
	{
		flmAssert( m_uiCursor < m_uiLength);
		
		return( m_pString[ m_uiCursor]);
	}


	/* Look for 'aByte' in the string, starting at the cursor.  If the byte if 
		found, then return TRUE.  Also return the offset of the byte relative
		to the start of the string */
	FINLINE FLMBOOL Scan(					//Scan for the presence of a given byte
		FLMBYTE			aByte,
		FLMUINT			&lLocation)
	{
		for( FLMUINT i = m_uiCursor; i < m_uiLength ; i++ )
		{
			if ( m_pString[i] == aByte )
			{
				lLocation = i;
				return( TRUE );
			}
		}
		return( FALSE );
	}


	enum GSD_SubState
	{
		GSD_Initial,						// No substring has been started
		GSD_BuildingASubString			// A substring has been started
	};
	
	
	FINLINE void MarkSubString(			// Mark a substring as Inserted or Deleted
		GSD_DifferenceType type)
	{
		if ( m_eSubState == GSD_BuildingASubString)
		{
			GSD_DifferenceData	difference;
			
			m_eSubState = GSD_Initial;
			MarkEndOfSubString();

			if ( m_uiSubLength )
			{
				// notify the caller that a difference has been found
				difference.type = type;
				difference.pSubString = m_pString + m_uiSubStart;
				difference.uiLength = m_uiSubLength;
				difference.uiPosition = m_uiSubStart;
				CallBack( difference );
			}			
		}
		/* Else there is no substring to mark, just return */
	}

	FINLINE void MarkStartOfSubString()//Start tracking a substring
	{
		if ( m_eSubState == GSD_Initial )
		{
			m_eSubState = GSD_BuildingASubString;
			m_uiSubStart = m_uiCursor;
		}
		/* Else a substring already exists, just return */
	}
	

private:

	FINLINE void MarkEndOfSubString()
	{
		m_uiSubLength = m_uiCursor - m_uiSubStart;
	}

	FINLINE void CallBack( 				// Call the caller's callback function
		GSD_DifferenceData	&difference)
	{
		(*m_pCallBack)( difference, m_pvCallBackData);
	}


	StringCursor(){}						// Not allowed

	FLMBYTE *			m_pString;		// Pointer to the string of bytes
	FLMUINT				m_uiLength;		// number of bytes in the string
	FLMUINT				m_uiCursor;		// offset of current byte in string

	/* Each StringCursor object can keep track of one substring */
	GSD_SubState		m_eSubState;	// the state of the substring
	FLMUINT				m_uiSubStart;	// offset of start of substring
	FLMUINT				m_uiSubLength;	// length of substring

	GSD_CallBackFunction	m_pCallBack;// Pointer to caller's callback function
	void *				m_pvCallBackData;// Pointer to caller's data
};

/****************************************************************************
Desc:		Find the differences between the 2 strings.  The strings are not
			assumed to be NULL-terminated, thus the required length parameter.
Notes:	This algorithm is intended to be accurate; however it does not always
			generate the smallest number of differences.  Like 
			flmRecordDifference, there cases where less than the optimal
			number of differences are reported.
****************************************************************************/
void flmStringDifference(
	FLMBYTE *			pBefore,
	FLMUINT				lBeforeLength,
	FLMBYTE *			pAfter,
	FLMUINT				lAfterLength,
	GSD_CallBackFunction	pCallBackFunction,
	void *				pvCallBackData)
{
	FLMUINT				uiPosition;
	StringCursor		beforeCursor( pBefore, lBeforeLength, pCallBackFunction,
								pvCallBackData);

	StringCursor		afterCursor( pAfter, lAfterLength, pCallBackFunction,
								pvCallBackData);

	
	// Iterate through all of the bytes in the 'before string'
	while ( ! beforeCursor.EndOfString() )
	{
		if ( afterCursor.EndOfString() )
		{	/* The end of the 'after string' has been reached, mark the remainder
				of the 'before string' as deleted */
			beforeCursor.MarkStartOfSubString();
			beforeCursor.AdvanceToEndOfString();
			beforeCursor.MarkSubString(GSD_Deleted);
			break;
		}

		if ( beforeCursor.Byte() == afterCursor.Byte() )
		{	// The 'before' and 'after' bytes are the same, advance both cursors
		
			/* If there is no substring to mark deleted, then this method
				invocation is a no-op */
			beforeCursor.MarkSubString(GSD_Deleted);
			
			beforeCursor.Advance();
			afterCursor.Advance();
		}
		else
		{	// The 'before' and 'after' bytes are different, log the difference
		
			//scan from afterCursor to EndOfString looking for 'before byte'
			if ( afterCursor.Scan( beforeCursor.Byte(), uiPosition ) )
			{	/* The 'before byte' was found in the 'after string' */

				/* If there is no substring to mark deleted, then this method
					invocation is a no-op */
				beforeCursor.MarkSubString(GSD_Deleted);
			
				/* Mark the preceding 'after bytes' as inserted */
				afterCursor.MarkStartOfSubString();
				afterCursor.Advance(uiPosition);
				afterCursor.MarkSubString(GSD_Inserted);
			}
			else
			{	/* The 'before byte' doesn't exist in the 'after string'  It must
					have been deleted.  Add the 'before byte' to a substring to be
					marked deleted. */

				/* If a substring has already been started, then this method
					invocation is a no-op */
				beforeCursor.MarkStartOfSubString();
				
				beforeCursor.Advance();
			}
		}
	}

	/* If there is no substring to mark deleted, then this method
		invocation is a no-op */
	beforeCursor.MarkSubString(GSD_Deleted);

	if ( ! afterCursor.EndOfString() )
	{	/* The end of the 'before string' has been reached, mark the remaining 
			'after bytes' as inserted */
		afterCursor.MarkStartOfSubString();
		afterCursor.AdvanceToEndOfString();
		afterCursor.MarkSubString(GSD_Inserted);
	}
}	// End of flmStringDifference

