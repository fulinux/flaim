//-------------------------------------------------------------------------
// Desc:	Query evaluation
// Tabs:	3
//
//		Copyright (c) 1994-2006 Novell, Inc. All Rights Reserved.
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
// $Id: fqeval.cpp 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#include "flaimsys.h"

FSTATIC FLMUINT flmCurEvalTrueFalse(
	FQATOM *				pElm);

FSTATIC RCODE flmCurGetAtomFromRec(
	FDB *					pDb,
	POOL *				pPool,
	FQATOM *				pTreeAtom,
	FlmRecord *			pRecord,
	QTYPES				eFldType,
	FLMBOOL				bGetAtomVals,
	FQATOM *				pResult,
	FLMBOOL				bHaveKey);
	
FSTATIC RCODE flmFieldIterate(
	FDB *					pDb,
	POOL *				pPool,
	QTYPES				eFldType,
	FQNODE *				pOpCB,
	FlmRecord *			pRecord,
	FLMBOOL				bHaveKey,
	FLMBOOL				bGetAtomVals,
	FLMUINT				uiAction,
	FQATOM *				pResult);

FSTATIC RCODE flmCurEvalArithOp(
	FDB *					pDb,
	SUBQUERY *			pSubQuery,
	FlmRecord *			pRecord,
	FQNODE *				pQNode,
	QTYPES				eOp,
	FLMBOOL				bGetNewField,
	FLMBOOL				bHaveKey,
	FQATOM *				pResult);

FSTATIC RCODE flmCurEvalLogicalOp(
	FDB *					pDb,
	SUBQUERY *			pSubQuery,
	FlmRecord *			pRecord,
	FQNODE *				pQNode,
	QTYPES				eOp,
	FLMBOOL				bHaveKey,
	FQATOM *				pResult);

FSTATIC RCODE OpSyntaxError(
	FQATOM *				pLhs,
	FQATOM * 			pRhs,
	FQATOM * 			pResult);

FSTATIC RCODE OpUUBitAND(
	FQATOM *				pLhs,
	FQATOM *				pRhs,
	FQATOM *				pResult);

FSTATIC RCODE OpUUBitOR(
	FQATOM *				pLhs,
	FQATOM *				pRhs,
	FQATOM *				pResult);

FSTATIC RCODE OpUUBitXOR(
	FQATOM *				pLhs,
	FQATOM *				pRhs,
	FQATOM *				pResult);

FSTATIC RCODE OpUUMult(
	FQATOM *				pLhs,
	FQATOM * 			pRhs,
	FQATOM * 			pResult);

FSTATIC RCODE OpUSMult(
	FQATOM *				pLhs,
	FQATOM *				pRhs,
	FQATOM *				pResult);

FSTATIC RCODE OpSSMult(
	FQATOM *				pLhs,
	FQATOM *				pRhs,
	FQATOM *				pResult);

FSTATIC RCODE OpSUMult(
	FQATOM *				pLhs,
	FQATOM *				pRhs,
	FQATOM *				pResult);

FSTATIC RCODE OpURMult(
	FQATOM * 			pLhs,
	FQATOM *				pRhs,
	FQATOM *				pResult);

FSTATIC RCODE OpRUMult(
	FQATOM *				pLhs,
	FQATOM *				pRhs,
	FQATOM *				pResult);

FSTATIC RCODE OpSRMult(
	FQATOM *				pLhs,
	FQATOM *				pRhs,
	FQATOM *				pResult);

FSTATIC RCODE OpRSMult(
	FQATOM *				pLhs,
	FQATOM *				pRhs,
	FQATOM *				pResult);

FSTATIC RCODE OpRRMult(
	FQATOM *				pLhs,
	FQATOM *				pRhs,
	FQATOM *				pResult);

FSTATIC RCODE OpUUDiv(
	FQATOM *				pLhs,
	FQATOM *				pRhs,
	FQATOM *				pResult);

FSTATIC RCODE OpUSDiv(
	FQATOM *				pLhs, 
	FQATOM *				pRhs,
	FQATOM *				pResult);

FSTATIC RCODE OpSSDiv(
	FQATOM *				pLhs,
	FQATOM *				pRhs,
	FQATOM *				pResult);

FSTATIC RCODE OpSUDiv(
	FQATOM *				pLhs,
	FQATOM *				pRhs,
	FQATOM *				pResult);

FSTATIC RCODE OpURDiv(
	FQATOM *				pLhs,
	FQATOM *				pRhs,
	FQATOM *				pResult);

FSTATIC RCODE OpRUDiv(
	FQATOM *				pLhs,
	FQATOM *				pRhs,
	FQATOM *				pResult);

FSTATIC RCODE OpSRDiv(
	FQATOM *				pLhs,
	FQATOM *				pRhs,
	FQATOM *				pResult);

FSTATIC RCODE OpRSDiv(
	FQATOM *				pLhs,
	FQATOM *				pRhs,
	FQATOM *				pResult);

FSTATIC RCODE OpRRDiv(
	FQATOM *				pLhs,
	FQATOM *				pRhs,
	FQATOM *				pResult);

FSTATIC RCODE OpUUMod(
	FQATOM *				pLhs,
	FQATOM *				pRhs,
	FQATOM *				pResult);

FSTATIC RCODE OpUSMod(
	FQATOM *				pLhs,
	FQATOM *				pRhs,
	FQATOM *				pResult);

FSTATIC RCODE OpSSMod(
	FQATOM *				pLhs,
	FQATOM *				pRhs,
	FQATOM *				pResult);

FSTATIC RCODE OpSUMod(
	FQATOM *				pLhs,
	FQATOM *				pRhs,
	FQATOM *				pResult);

FSTATIC RCODE OpUUPlus(
	FQATOM *				pLhs,
	FQATOM *				pRhs,
	FQATOM *				pResult);

FSTATIC RCODE OpUSPlus(
	FQATOM *				pLhs,
	FQATOM *				pRhs,
	FQATOM *				pResult);

FSTATIC RCODE OpSSPlus(
	FQATOM *				pLhs,
	FQATOM *				pRhs,
	FQATOM *				pResult);

FSTATIC RCODE OpSUPlus(
	FQATOM *				pLhs,
	FQATOM *				pRhs,
	FQATOM *				pResult);

FSTATIC RCODE OpURPlus(
	FQATOM *				pLhs,
	FQATOM *				pRhs,
	FQATOM *				pResult);

FSTATIC RCODE OpRUPlus(
	FQATOM *				pLhs,
	FQATOM *				pRhs,
	FQATOM *				pResult);

FSTATIC RCODE OpSRPlus(
	FQATOM *				pLhs,
	FQATOM *				pRhs,
	FQATOM *				pResult);

FSTATIC RCODE OpRSPlus(
	FQATOM *				pLhs,
	FQATOM *				pRhs,
	FQATOM *				pResult);

FSTATIC RCODE OpRRPlus(
	FQATOM *				pLhs,
	FQATOM *				pRhs,
	FQATOM *				pResult);

FSTATIC RCODE OpUUMinus(
	FQATOM *				pLhs,
	FQATOM *				pRhs,
	FQATOM *				pResult);

FSTATIC RCODE OpUSMinus(
	FQATOM *				pLhs,
	FQATOM *				pRhs,
	FQATOM *				pResult);

FSTATIC RCODE OpSSMinus(
	FQATOM *				pLhs,
	FQATOM *				pRhs,
	FQATOM *				pResult);

FSTATIC RCODE OpSUMinus(
	FQATOM *				pLhs,
	FQATOM *				pRhs,
	FQATOM *				pResult);

FSTATIC RCODE OpURMinus(
	FQATOM *				pLhs,
	FQATOM *				pRhs,
	FQATOM *				pResult);

FSTATIC RCODE OpRUMinus(
	FQATOM *				pLhs,
	FQATOM *				pRhs,
	FQATOM *				pResult);

FSTATIC RCODE OpSRMinus(
	FQATOM *				pLhs,
	FQATOM *				pRhs,
	FQATOM *				pResult);

FSTATIC RCODE OpRSMinus(
	FQATOM *				pLhs,
	FQATOM *				pRhs,
	FQATOM *				pResult);

FSTATIC RCODE OpRRMinus(
	FQATOM *				pLhs,
	FQATOM *				pRhs,
	FQATOM *				pResult);

#define IS_EXPORT_PTR(e) \
	((e) == FLM_TEXT_VAL || (e) == FLM_BINARY_VAL)

/****************************************************************************
Desc:
****************************************************************************/
FQ_OPERATION * FQ_DoOperation[((LAST_ARITH_OP - FIRST_ARITH_OP) + 1) * 9] =
{
	// BITAND
	
	OpUUBitAND,
	OpUUBitAND,
	OpSyntaxError,
	OpUUBitAND,
	OpUUBitAND,
	OpSyntaxError,
	OpSyntaxError,
	OpSyntaxError,
	OpSyntaxError,

	// BITOR

	OpUUBitOR,
	OpUUBitOR,
	OpSyntaxError,
	OpUUBitOR,
	OpUUBitOR,
	OpSyntaxError,
	OpSyntaxError,
	OpSyntaxError,
	OpSyntaxError,

	// BITXOR

	OpUUBitXOR,
	OpUUBitXOR,
	OpSyntaxError,
	OpUUBitXOR,
	OpUUBitXOR,
	OpSyntaxError,
	OpSyntaxError,
	OpSyntaxError,
	OpSyntaxError,

	// MULT

	OpUUMult,
	OpUSMult,
	OpURMult,
	OpSUMult,
	OpSSMult,
	OpSRMult,
	OpRUMult,
	OpRSMult,
	OpRRMult,

	// DIV

	OpUUDiv,
	OpUSDiv,
	OpURDiv,
	OpSUDiv,
	OpSSDiv,
	OpSRDiv,
	OpRUDiv,
	OpRSDiv,
	OpRRDiv,

	// MOD

	OpUUMod,
	OpUSMod,
	OpSyntaxError,
	OpSUMod,
	OpSSMod,
	OpSyntaxError,
	OpSyntaxError,
	OpSyntaxError,
	OpSyntaxError,

	// PLUS

	OpUUPlus,
	OpUSPlus,
	OpURPlus,
	OpSUPlus,
	OpSSPlus,
	OpSRPlus,
	OpRUPlus,
	OpRSPlus,
	OpRRPlus,

	// MINUS

	OpUUMinus,
	OpUSMinus,
	OpURMinus,
	OpSUMinus,
	OpSSMinus,
	OpSRMinus,
	OpRUMinus,
	OpRSMinus,
	OpRRMinus
};

/****************************************************************************
Desc: Evaluates a list of QATOM elements, and returns a complex boolean
		based on their contents.
Ret:	FLM_TRUE if all elements have nonzero numerics or nonempty buffers.
		FLM_FALSE if all contents are zero or empty. 
		FLM_UNK if any QATOM is of type FLM_UNKNOWN.
		Any combination of the preceeding values if their corresponding 
		criteria are met.
****************************************************************************/
FSTATIC FLMUINT flmCurEvalTrueFalse(
	FQATOM *	pQAtom)
{
	FQATOM *	pTmpQAtom;
	FLMUINT	uiTrueFalse = 0;

	for (pTmpQAtom = pQAtom; pTmpQAtom; pTmpQAtom = pTmpQAtom->pNext)
	{
		if (IS_BUF_TYPE( pTmpQAtom->eType))
		{
			if (pTmpQAtom->uiBufLen > 0)
			{
				uiTrueFalse |= FLM_TRUE;
			}
			else
			{
				uiTrueFalse |= FLM_FALSE;
			}
		}
		else
		{
			switch (pTmpQAtom->eType)
			{
				case FLM_BOOL_VAL:
					uiTrueFalse |= pTmpQAtom->val.uiBool;
					break;
				case FLM_UNKNOWN:
					uiTrueFalse |= FLM_UNK;
					break;
				case FLM_INT32_VAL:
					if (pTmpQAtom->val.iVal)
					{
						uiTrueFalse |= FLM_TRUE;
					}
					else
					{
						uiTrueFalse |= FLM_FALSE;
					}
					break;
				case FLM_UINT32_VAL:
					if (pTmpQAtom->val.uiVal)
					{
						uiTrueFalse |= FLM_TRUE;
					}
					else
					{
						uiTrueFalse |= FLM_FALSE;
					}
					break;
				default:
					goto Exit;
			}
		}

		if (uiTrueFalse == FLM_ALL_BOOL)
		{
			break;
		}
	}

Exit:

	return (uiTrueFalse);
}

/****************************************************************************
Desc:	Gets a value from the passed-in record field and stuffs it into the
		passed-in FQATOM.
****************************************************************************/
RCODE flmCurGetAtomVal(
	FlmRecord *		pRecord,
	void *			pField,
	POOL *			pPool,
	QTYPES			eFldType,
	FQATOM *			pResult)
{
	RCODE		rc = FERR_OK;
	FLMUINT	uiType = 0;

	if (pField)
	{
		uiType = pRecord->getDataType( pField);
		if (uiType == FLM_BLOB_TYPE)
		{
			rc = RC_SET( FERR_NOT_IMPLEMENTED);
			goto Exit;
		}
	}

	switch (eFldType)
	{
		case FLM_TEXT_VAL:
		{
			if (!pField)
			{
				// Default value
				
				pResult->uiBufLen = 0;
				pResult->val.pucBuf = NULL;
			}
			else
			{
				pResult->uiBufLen = pRecord->getDataLength( pField);
				if (pResult->uiBufLen)
				{
					pResult->val.pucBuf = (FLMBYTE *) pRecord->getDataPtr( pField);
					pResult->pFieldRec = pRecord;
				}
				else
				{
					if ((pResult->val.pucBuf = (FLMBYTE *) GedPoolAlloc( 
						pPool, 1)) == NULL)
					{
						rc = RC_SET( FERR_MEM);
						break;
					}

					pResult->val.pucBuf[0] = 0;
				}
			}

			pResult->eType = FLM_TEXT_VAL;
			break;
		}
		
		case FLM_INT32_VAL:
		{
			if (!pField || pRecord->getDataLength( pField) == 0)
			{
				// Default value
				
				pResult->val.iVal = 0;
			}
			else if (uiType == FLM_NUMBER_TYPE || uiType == FLM_TEXT_TYPE)
			{
				if (RC_BAD( rc = pRecord->getINT( pField, &pResult->val.iVal)))
				{

					// Try to get the number as an unsigned value. For purposes
					// of evaluation, the 32-bit value will still be treated as
					// signed. In effect, the large positive value is wrapped and
					// becomes a negative value.

					if (rc == FERR_CONV_NUM_OVERFLOW)
					{
						rc = pRecord->getUINT( pField, &pResult->val.uiVal);
						eFldType = FLM_UINT32_VAL;
					}
				}
			}
			else if (uiType == FLM_CONTEXT_TYPE)
			{
				rc = pRecord->getUINT( pField, &pResult->val.uiVal);
				eFldType = FLM_UINT32_VAL;
			}
			else
			{
				rc = RC_SET( FERR_CONV_BAD_SRC_TYPE);
			}

			if (RC_OK( rc))
			{
				pResult->eType = eFldType;
			}
			break;
		}
		
		case FLM_UINT32_VAL:
		case FLM_REC_PTR_VAL:
		{
			if (!pField || pRecord->getDataLength( pField) == 0)
			{
				// Default value
				
				pResult->val.uiVal = 0;
			}
			else if (uiType == FLM_NUMBER_TYPE || uiType == FLM_TEXT_TYPE)
			{
				if (RC_BAD( rc = pRecord->getUINT( pField, &pResult->val.uiVal)))
				{

					// Try to get the number as a signed value. For purposes of
					// evaluation, the 32-bit value will still be treated as
					// unsigned. In effect, the negative value is wrapped and
					// becomes a large positive value.

					if (rc == FERR_CONV_NUM_UNDERFLOW)
					{
						rc = pRecord->getINT( pField, &pResult->val.iVal);
						eFldType = FLM_INT32_VAL;
					}
				}
			}
			else if (uiType == FLM_CONTEXT_TYPE)
			{
				rc = pRecord->getUINT( pField, &(pResult->val.uiVal));
			}
			else
			{
				rc = RC_SET( FERR_CONV_BAD_SRC_TYPE);
			}

			if (RC_OK( rc))
			{
				pResult->eType = eFldType;
			}
			break;
		}
		
		case FLM_BINARY_VAL:
		{
			if (pField)
			{
				pResult->uiBufLen = pRecord->getDataLength( pField);
			}
			else
			{
				pResult->uiBufLen = 0;
			}

			if (!pResult->uiBufLen)
			{
				pResult->val.pucBuf = NULL;
			}
			else
			{
				pResult->val.pucBuf = (FLMBYTE *) pRecord->getDataPtr( pField);
				pResult->pFieldRec = pRecord;
			}

			pResult->eType = FLM_BINARY_VAL;
			break;
		}

		// No type -- use the type in the passed-in node.

		case NO_TYPE:
		{

			// At this point, if we are attempting to get a default value, but
			// don't know the type, it is because both sides of the operand are
			// unknown, so we need to return no type.

			if (!pField)
			{
				pResult->eType = NO_TYPE;
			}
			else
			{
				switch (uiType)
				{
					case FLM_TEXT_TYPE:
					{
						pResult->uiBufLen = pRecord->getDataLength( pField);
						if (pResult->uiBufLen)
						{
							pResult->val.pucBuf = (FLMBYTE *) pRecord->getDataPtr( pField);
							pResult->pFieldRec = pRecord;
						}
						else
						{
							if ((pResult->val.pucBuf = (FLMBYTE *) GedPoolAlloc( 
								pPool, 1)) == NULL)
							{
								rc = RC_SET( FERR_MEM);
								break;
							}

							pResult->val.pucBuf[0] = 0;
						}

						pResult->eType = FLM_TEXT_VAL;
						break;
					}

					case FLM_BINARY_TYPE:
					{
						if (pField)
						{
							pResult->uiBufLen = pRecord->getDataLength( pField);
						}
						else
						{
							pResult->uiBufLen = 0;
						}
	
						if (!pResult->uiBufLen)
						{
							pResult->val.pucBuf = NULL;
						}
						else
						{
							pResult->val.pucBuf = 
								(FLMBYTE *) pRecord->getDataPtr( pField);
							pResult->pFieldRec = pRecord;
						}
	
						pResult->eType = FLM_BINARY_VAL;
						break;
					}
					
					case FLM_NUMBER_TYPE:
					{
						if (RC_OK( rc = pRecord->getUINT( pField, 
								&pResult->val.uiVal)))
						{
							pResult->eType = FLM_UINT32_VAL;
						}
						else if (RC_OK( rc = pRecord->getINT( pField, 
								&pResult->val.iVal)))
						{
							pResult->eType = FLM_INT32_VAL;
						}
						break;
					}
					
					case FLM_CONTEXT_TYPE:
					{
						if (RC_OK( rc = pRecord->getUINT( pField, 
								&(pResult->val.uiVal))))
						{
							pResult->eType = FLM_UINT32_VAL;
						}
						break;
					}
				}
			}
			break;
		}
		
		default:
		{
			rc = RC_SET( FERR_CURSOR_SYNTAX);
			break;
		}
	}

Exit:

	pResult->uiFlags &= 
		~(FLM_IS_RIGHT_TRUNCATED_DATA | FLM_IS_LEFT_TRUNCATED_DATA);
		
	if (RC_OK( rc) && pField)
	{
		if (pRecord->isRightTruncated( pField))
		{
			pResult->uiFlags |= FLM_IS_RIGHT_TRUNCATED_DATA;
		}

		if (pRecord->isLeftTruncated( pField))
		{
			pResult->uiFlags |= FLM_IS_LEFT_TRUNCATED_DATA;
		}
	}

	return (rc);
}

/****************************************************************************
Desc: Given a list of FQATOMs containing alternate field paths, finds
		those field paths in a compound record and creates a list of FQATOMs 
		from the contents of those paths.
****************************************************************************/
FSTATIC RCODE flmCurGetAtomFromRec(
	FDB *				pDb,
	POOL *			pPool,
	FQATOM *			pTreeAtom,
	FlmRecord *		pRecord,
	QTYPES			eFldType,
	FLMBOOL			bGetAtomVals,
	FQATOM *			pResult,
	FLMBOOL			bHaveKey)
{
	RCODE				rc = FERR_OK;
	FQATOM *			pTmpResult = NULL;
	void *			pvField;
	void *			pvLastLevelOneField;
	FLMUINT *		puiFldPath;
	FLMUINT			uiCurrFieldPath[ GED_MAXLVLNUM + 1];
	FLMUINT			uiFieldLevel;
	FLMUINT			uiTmp;
	FLMUINT			uiLeafFldNum;
	FLMUINT			uiRecFldNum;
	FLMBOOL			bFound;
	FLMBOOL			bSavedInvisTrans;
	FLMUINT			uiResult;
	FLMBOOL			bPathFromRoot;
	FLMBOOL			bUseFieldIdLookupTable;
	FLMUINT *		puiPToCPath;
	FLMUINT			uiHighestLevel;
	FLMUINT			uiLevelOneFieldId;

	pResult->eType = NO_TYPE;
	if (pTreeAtom->val.QueryFld.puiFldPath [0] == FLM_MISSING_FIELD_TAG)
	{
		goto Exit;
	}
	
	if (!pRecord)
	{
		goto Exit;
	}

	flmAssert( !pTreeAtom->pNext);
	puiFldPath = pTreeAtom->val.QueryFld.puiFldPath;
	puiPToCPath = pTreeAtom->val.QueryFld.puiPToCPath;
	uiLevelOneFieldId = puiPToCPath [1];
	
	// We are only going to do the path to root optimation if
	// the field path is specified as having to be from the root (FLM_ROOTED_PATH)
	// and it goes down to at least level 1 in the tree, and our record
	// has a field id table in it.
	
	bPathFromRoot = (!bHaveKey &&
						  (pTreeAtom->uiFlags & FLM_ROOTED_PATH))
						 ? TRUE
						 : FALSE;
						 
	bUseFieldIdLookupTable = (bPathFromRoot &&
									  pRecord->fieldIdTableEnabled() &&
									  uiLevelOneFieldId)
									 ? TRUE
									 : FALSE;
									 
	if (*puiFldPath == FLM_RECID_FIELD)
	{
		pResult->eType = FLM_UINT32_VAL;
		pResult->val.uiVal = pRecord->getID();
		goto Exit;
	}
	
	pvField = pRecord->root();
	uiFieldLevel = 0;
	
	if (bPathFromRoot)
	{
		// Determine the highest level we need to go down to in the record.
		
		uiHighestLevel = 1;
		while (puiPToCPath [uiHighestLevel + 1])
		{
			uiHighestLevel++;
		}
		
		if (puiPToCPath [0] != pRecord->getFieldID( pvField))
		{
			goto Exit;
		}
		
		if (bUseFieldIdLookupTable)
		{
			if ((pvLastLevelOneField =
					pRecord->findLevelOneField( uiLevelOneFieldId, FALSE)) == NULL)
			{
				goto Exit;
			}
			
			uiCurrFieldPath [0] = puiPToCPath [0];
			pvField = pvLastLevelOneField;
			uiFieldLevel = 1;
		}
	}
	
	uiLeafFldNum = puiFldPath[ 0];
	for (;;)
	{
		uiRecFldNum = pRecord->getFieldID( pvField);
		uiCurrFieldPath[ uiFieldLevel] = uiRecFldNum;
		
		// When we are doing path from root, we only need to traverse
		// back up when we are on a field that is exactly at the highest level
		// we can go down to in the tree - no need to check any others.
		// If we are not doing bPathFromRoot, we check all node paths.

		if (uiRecFldNum == uiLeafFldNum &&
			 (!bPathFromRoot || uiFieldLevel == uiHighestLevel))
		{
			bFound = TRUE;
			
			// We already know that puiFldPath[0] matches - it is the same
			// as uiLeafFldNum.  Traverse back up the tree and see if
			// the rest of the path matches.
			
			for (uiTmp = 1; puiFldPath[ uiTmp]; uiTmp++)
			{
				if (!uiFieldLevel)
				{
					bFound = FALSE;
					break;
				}
				
				uiFieldLevel--;
				
				if (puiFldPath[ uiTmp] != uiCurrFieldPath[ uiFieldLevel])
				{
					bFound = FALSE;
					break;
				}
			}

			// Found field in proper path.  Get the value if requested,
			// otherwise set the result to FLM_TRUE and exit.  If a
			// callback is set, do that first to see if it is REALLY
			// found.

			if (bFound && pTreeAtom->val.QueryFld.fnGetField)
			{
				CB_ENTER( pDb, &bSavedInvisTrans);
				rc = pTreeAtom->val.QueryFld.fnGetField(
							pTreeAtom->val.QueryFld.pvUserData, pRecord,
							(HFDB)pDb, pTreeAtom->val.QueryFld.puiFldPath,
							FLM_FLD_VALIDATE, NULL, &pvField, &uiResult);
				CB_EXIT( pDb, bSavedInvisTrans);
				
				if (RC_BAD( rc))
				{
					goto Exit;
				}
				
				if (uiResult == FLM_FALSE)
				{
					bFound = FALSE;
				}
				else if (uiResult == FLM_UNK)
				{
					if (bHaveKey)
					{

						// bHaveKey means we are evaluating a key.  There
						// should only be one occurrence of the field in the
						// key in this case.  If the callback does not know
						// if the field really exists, we must defer judgement
						// on this one until we can fetch the record.  Hence,
						// we force the result to be UNKNOWN.  Note that it
						// must be set to UNKNOWN, even if this is a field exists
						// predicate (!bGetAtomVals).  If we set it to NO_TYPE
						// and fall through to exist, it would get converted to
						// a FLM_BOOL_VAL of FALSE, which is NOT what we
						// want.

						pResult->eType = FLM_UNKNOWN;
						pResult->uiFlags = pTreeAtom->uiFlags &
													~(FLM_IS_RIGHT_TRUNCATED_DATA |
													  FLM_IS_LEFT_TRUNCATED_DATA);
						if (pvField)
						{
							if (pRecord->isRightTruncated( pvField))
							{
								pResult->uiFlags |= FLM_IS_RIGHT_TRUNCATED_DATA;
							}
							if (pRecord->isLeftTruncated( pvField))
							{
								pResult->uiFlags |= FLM_IS_LEFT_TRUNCATED_DATA;
							}
						}

						// Better not be multiple results in this case because
						// we are evaluating a key.

						flmAssert( pResult->pNext == NULL);
						pResult->pNext = NULL;
						goto Exit;
					}
					else
					{
						bFound = FALSE;
					}
				}
			}

			if (bFound)
			{
				if (!bGetAtomVals)
				{
					pResult->eType = FLM_BOOL_VAL;
					pResult->val.uiBool = FLM_TRUE;
					goto Exit;
				}
				
				if (!pTmpResult)
				{
					pTmpResult = pResult;
				}
				else if (pTmpResult->eType)
				{
					if ((pTmpResult->pNext =
						(FQATOM *)GedPoolCalloc( pPool, sizeof( FQATOM))) == NULL)
					{
						rc = RC_SET( FERR_MEM);
						goto Exit;
					}
					pTmpResult = pTmpResult->pNext;
				}
				
				pTmpResult->uiFlags = pTreeAtom->uiFlags;
				if ((rc = flmCurGetAtomVal( pRecord, pvField, pPool, eFldType,
									pTmpResult)) == FERR_CURSOR_SYNTAX)
				{
					goto Exit;
				}
			}
		}
		
		// Get the next field to process.  If bPathFromRoot is set, we will skip
		// any fields that are at too high of levels in the record.
		// If bUseFieldIdLookupTable is set, it means
		// that when we get back up to level one fields, we should call the
		// API to get the next level one field.
		
		for (;;)
		{
			if ((pvField = pRecord->next( pvField)) == NULL)
			{
				break;
			}
			
			uiFieldLevel = pRecord->getLevel( pvField);
			
			if (!bPathFromRoot)
			{
				break;
			}
			
			if (uiFieldLevel > uiHighestLevel)
			{
				continue;
			}
			
			if (bUseFieldIdLookupTable && uiFieldLevel == 1)
			{
				pvLastLevelOneField = pvField = pRecord->nextLevelOneField(
															pvLastLevelOneField);
			}
			
			break;
		}
		
		// If the end of the record has been reached, and the last field
		// value searched for was not found, unlink it from the result list.

		if (!pvField)
		{
			if (pTmpResult && pTmpResult != pResult &&
				 pTmpResult->eType == NO_TYPE)
			{
				FQATOM *		pTmp;

				for (pTmp = pResult;
					  pTmp && pTmp->pNext != pTmpResult;
					  pTmp = pTmp->pNext)
				{
					;
				}
				pTmp->pNext = NULL;
			}
			break;
		}
	}
	
Exit:

	// If no match was found anywhere, set the result to FLM_UNKNOWN if field
	// content was requested, or FLM_FALSE if field existence was to be tested.

	if (pResult->eType == NO_TYPE)
	{
		if (bGetAtomVals && !bHaveKey &&
			!pTreeAtom->val.QueryFld.fnGetField &&
			(pTreeAtom->uiFlags & FLM_USE_DEFAULT_VALUE))
		{
			rc = flmCurGetAtomVal( pRecord, NULL, pPool, eFldType, pResult);
		}
		else
		{
			if (bGetAtomVals || bHaveKey)
			{
				pResult->eType = FLM_UNKNOWN;
			}
			else
			{
				pResult->eType = FLM_BOOL_VAL;
				pResult->val.uiBool = FLM_FALSE;
			}
			pResult->uiFlags = pTreeAtom->uiFlags;
		}
	}
	
	return( rc);
}

/****************************************************************************
Desc: Iterate to the next occurrance of a field.
****************************************************************************/
FSTATIC RCODE flmFieldIterate(
	FDB *				pDb,
	POOL *			pPool,
	QTYPES			eFldType,
	FQNODE *			pOpCB,
	FlmRecord *		pRecord,
	FLMBOOL			bHaveKey,
	FLMBOOL			bGetAtomVals,
	FLMUINT			uiAction,
	FQATOM *			pResult)
{
	RCODE				rc = FERR_OK;
	FlmRecord *		pFieldRec = NULL;
	void *			pField = NULL;
	FLMBOOL			bSavedInvisTrans;

	if (bHaveKey)
	{

		// bHaveKey is TRUE when we are evaluating a key instead of the
		// full record. In this case, it will not be possible for the
		// callback function to get all of the values - so we simply return
		// unknown, which will be handled by the outside. If the entire
		// query evaluates to unknown, FLAIM will fetch the record and
		// evaluate the entire thing. This is the safe route to take in this
		// case.

		pResult->eType = FLM_UNKNOWN;
	}
	else
	{
		CB_ENTER( pDb, &bSavedInvisTrans);
		rc = pOpCB->pQAtom->val.QueryFld.fnGetField( 
			pOpCB->pQAtom->val.QueryFld.pvUserData, pRecord, (HFDB) pDb,
			pOpCB->pQAtom->val.QueryFld.puiFldPath, uiAction, 
			&pFieldRec, &pField, NULL);
		CB_EXIT( pDb, bSavedInvisTrans);

		if (RC_BAD( rc))
		{
			goto Exit;
		}

		if (!pField)
		{
			if (!bGetAtomVals)
			{
				pResult->eType = FLM_BOOL_VAL;
				pResult->val.uiBool = FLM_FALSE;
			}
			else
			{
				if ((pOpCB->pQAtom->uiFlags & FLM_USE_DEFAULT_VALUE) &&
					 (uiAction == FLM_FLD_FIRST))
				{
					if (RC_BAD( rc = flmCurGetAtomVal( pFieldRec, NULL, pPool,
								  eFldType, pResult)))
					{
						goto Exit;
					}
				}
				else
				{
					pResult->eType = FLM_UNKNOWN;
				}
			}
		}
		else
		{
			if (!bGetAtomVals)
			{
				pResult->eType = FLM_BOOL_VAL;
				pResult->val.uiBool = FLM_TRUE;
			}
			else if (RC_BAD( rc = flmCurGetAtomVal( pFieldRec, pField, pPool,
								 eFldType, pResult)))
			{
				goto Exit;
			}
		}
	}

Exit:

	return (rc);
}

/****************************************************************************
Desc:	Performs arithmetic operations on stack element lists.
****************************************************************************/
FSTATIC RCODE flmCurEvalArithOp(
	FDB *			pDb,
	SUBQUERY *	pSubQuery,
	FlmRecord *	pRecord,
	FQNODE *		pQNode,
	QTYPES		eOp,
	FLMBOOL		bGetNewField,
	FLMBOOL		bHaveKey,
	FQATOM *		pResult)
{
	RCODE			rc = FERR_OK;
	FQNODE *		pTmpQNode;
	FQATOM		Lhs;
	FQATOM		Rhs;
	FQATOM *		pTmpQAtom;
	FQATOM *		pRhs;
	FQATOM *		pLhs;
	FQATOM *		pFirstRhs;
	QTYPES		eType;
	QTYPES		eFldType = NO_TYPE;
	FLMBOOL		bSecondOperand = FALSE;
	FQNODE *		pRightOpCB = NULL;
	FQNODE *		pLeftOpCB = NULL;
	FQNODE *		pOpCB = NULL;
	POOL *		pTmpPool = &pDb->TempPool;
	FLMBOOL		bSavedInvisTrans;
	RCODE			TempRc;

	if ((pTmpQNode = pQNode->pChild) == NULL)
	{
		rc = RC_SET( FERR_CURSOR_SYNTAX);
		return (rc);
	}

	pLhs = &Lhs;
	pRhs = &Rhs;

	pLhs->pNext = NULL;
	pLhs->pFieldRec = NULL;
	pLhs->eType = NO_TYPE;
	pLhs->uiBufLen = 0;
	pLhs->val.uiVal = 0;

	pRhs->pNext = NULL;
	pRhs->pFieldRec = NULL;
	pRhs->eType = NO_TYPE;
	pRhs->uiBufLen = 0;
	pRhs->val.uiVal = 0;

	// Get the two operands (may be multiple values per operand)

	pTmpQAtom = pLhs;
	
Get_Operand:

	eType = GET_QNODE_TYPE( pTmpQNode);
	if (IS_FLD_CB( eType, pTmpQNode))
	{
		eType = FLM_CB_FLD;
	}

	if (IS_VAL( eType))
	{
		if (bSecondOperand)
		{
			pRhs = pTmpQNode->pQAtom;
		}
		else
		{
			pLhs = pTmpQNode->pQAtom;
		}
	}
	else if (eType == FLM_FLD_PATH || eType == FLM_CB_FLD)
	{
		if (bSecondOperand)
		{
			eFldType = pLhs->eType;
			if (eType == FLM_CB_FLD)
			{
				pOpCB = pRightOpCB = pTmpQNode;
			}
		}
		else
		{
			if (pTmpQNode->pNextSib == NULL)
			{
				rc = RC_SET( FERR_CURSOR_SYNTAX);
				goto Exit;
			}

			eFldType = GET_QNODE_TYPE( pTmpQNode->pNextSib);

			if (eType == FLM_CB_FLD)
			{
				pOpCB = pLeftOpCB = pTmpQNode;
			}
		}

		if (!IS_VAL( eFldType))
		{
			eFldType = NO_TYPE;
		}

		if (eType == FLM_CB_FLD)
		{

			// Get the first occurrence of the field.

			if (RC_BAD( rc = flmFieldIterate( pDb, pTmpPool, eFldType, pOpCB,
						  pRecord, bHaveKey, TRUE, FLM_FLD_FIRST, pTmpQAtom)))
			{
				goto Exit;
			}
		}
		else
		{
			if (RC_BAD( rc = flmCurGetAtomFromRec( pDb, pTmpPool,
						  pTmpQNode->pQAtom, pRecord, eFldType, TRUE, pTmpQAtom,
						  bHaveKey)))
			{
				goto Exit;
			}
		}
	}
	else if (IS_ARITH_OP( eType))
	{

		// Recursive call

		if (RC_BAD( rc = flmCurEvalArithOp( pDb, pSubQuery, pRecord, pTmpQNode,
					  eType, bGetNewField, bHaveKey, pTmpQAtom)))
		{
			goto Exit;
		}
	}
	else
	{
		rc = RC_SET( FERR_CURSOR_SYNTAX);
		goto Exit;
	}

	if (!bSecondOperand)
	{
		if (eOp == FLM_NEG_OP)
		{
			pResult = pTmpQAtom;
			flmCurDoNeg( pResult);
			goto Exit;
		}
		else
		{
			if (pTmpQNode->pNextSib == NULL)
			{
				rc = RC_SET( FERR_CURSOR_SYNTAX);
				goto Exit;
			}

			pTmpQNode = pTmpQNode->pNextSib;
			pTmpQAtom = pRhs;
			bSecondOperand = TRUE;
			goto Get_Operand;
		}
	}

	// Now do the operation using our operators

	pFirstRhs = pRhs;
	pTmpQAtom = pResult;

	for (;;)
	{
		if (pLhs->eType == FLM_UNKNOWN || pRhs->eType == FLM_UNKNOWN)
		{
			pTmpQAtom->eType = FLM_UNKNOWN;
		}
		else
		{
			int		opPos = (eOp - FIRST_ARITH_OP) * 9;
			QTYPES	lhType = (QTYPES) (pLhs->eType - FLM_UINT32_VAL);
			QTYPES	rhType = (QTYPES) (pRhs->eType - FLM_UINT32_VAL);

			if (lhType > 2 || rhType > 2)
			{
				rc = RC_SET( FERR_CURSOR_SYNTAX);
				goto Exit;
			}

			opPos += (lhType * 3) + rhType;

			// Call through a table the operation handling routine.

			rc = (FQ_DoOperation[opPos]) (pLhs, pRhs, pTmpQAtom);
		}

		// Doing contextless, do them all - loop through right hand
		// operands, then left hand operands.
		//
		// Get the next right hand operand.
		
		if (!pRightOpCB)
		{
			pRhs = pRhs->pNext;
		}
		else if (pRhs->eType == FLM_UNKNOWN)
		{
			pRhs = NULL;
		}
		else
		{
			if (RC_BAD( rc = flmFieldIterate( pDb, pTmpPool, eFldType, pRightOpCB,
						  pRecord, bHaveKey, TRUE, FLM_FLD_NEXT, pRhs)))
			{
				goto Exit;
			}

			if (pRhs->eType == FLM_UNKNOWN)
			{
				pRhs = NULL;
			}
		}

		// If no more right hand side, get the next left hand side, and
		// reset the right hand side.

		if (!pRhs)
		{
			if (!pLeftOpCB)
			{
				pLhs = pLhs->pNext;
			}
			else if (pLhs->eType == FLM_UNKNOWN)
			{
				pLhs = NULL;
			}
			else
			{
				if (RC_BAD( rc = flmFieldIterate( pDb, pTmpPool, eFldType,
							  pLeftOpCB, pRecord, bHaveKey, TRUE, FLM_FLD_NEXT, pLhs)))
				{
					goto Exit;
				}

				if (pLhs->eType == FLM_UNKNOWN)
				{
					pLhs = NULL;
				}
			}

			if (!pLhs)
			{
				break;
			}

			// Reset the right hand side back to first.

			if (pRightOpCB)
			{
				if (RC_BAD( rc = flmFieldIterate( pDb, pTmpPool, eFldType,
							  pRightOpCB, pRecord, bHaveKey, TRUE, FLM_FLD_FIRST, pRhs
							  )))
				{
					goto Exit;
				}
			}
			else
			{
				pRhs = pFirstRhs;
			}
		}

		// Set up for next result

		if ((pTmpQAtom->pNext = (FQATOM *) GedPoolCalloc( 
				pTmpPool, sizeof( FQATOM))) == NULL)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}

		pTmpQAtom = pTmpQAtom->pNext;
	}

Exit:

	// Clean up any field callbacks.

	if (pLeftOpCB)
	{
		CB_ENTER( pDb, &bSavedInvisTrans);
		TempRc = pLeftOpCB->pQAtom->val.QueryFld.fnGetField( 
			pLeftOpCB->pQAtom->val.QueryFld.pvUserData, NULL, (HFDB) pDb,
			pLeftOpCB->pQAtom->val.QueryFld.puiFldPath, FLM_FLD_RESET, NULL,
			NULL, NULL);
		CB_EXIT( pDb, bSavedInvisTrans);

		if (RC_BAD( TempRc))
		{
			if (RC_OK( rc))
			{
				rc = TempRc;
			}
		}
	}

	if (pRightOpCB)
	{
		CB_ENTER( pDb, &bSavedInvisTrans);
		TempRc = pRightOpCB->pQAtom->val.QueryFld.fnGetField( 
			pRightOpCB->pQAtom->val.QueryFld.pvUserData, NULL, (HFDB) pDb,
			pRightOpCB->pQAtom->val.QueryFld.puiFldPath, FLM_FLD_RESET, NULL,
			NULL, NULL);
		CB_EXIT( pDb, bSavedInvisTrans);

		if (RC_BAD( TempRc))
		{
			if (RC_OK( rc))
			{
				rc = TempRc;
			}
		}
	}

	return (rc);
}

/****************************************************************************
Desc:	Performs a comparison operation on two operands, one or both of
		which can be FLM_UNKNOWN.
****************************************************************************/
void flmCompareOperands(
	FLMUINT		uiLang,
	FQATOM *		pLhs,
	FQATOM *		pRhs,
	QTYPES		eOp,
	FLMBOOL		bResolveUnknown,
	FLMBOOL		bForEvery,
	FLMBOOL		bNotted,
	FLMBOOL		bHaveKey,
	FLMUINT *	puiTrueFalse)
{
	if (pLhs->eType == FLM_UNKNOWN || pRhs->eType == FLM_UNKNOWN)
	{

		// If we are not resolving predicates with unknown operands, return
		// FLM_UNK.

		if (bHaveKey || !bResolveUnknown)
		{
			*puiTrueFalse = FLM_UNK;
		}
		else if (bNotted)
		{

			// If bNotted is TRUE, the result will be inverted on the
			// outside, so we need to set it to the opposite of what we want
			// it to ultimately be.

			*puiTrueFalse = (bForEvery ? FLM_FALSE : FLM_TRUE);
		}
		else
		{
			*puiTrueFalse = (bForEvery ? FLM_TRUE : FLM_FALSE);
		}
	}

	// At this point, both operands are known to be present. The
	// comparison will therefore be performed according to the operator
	// specified.

	else
	{
		switch (eOp)
		{
			case FLM_EQ_OP:
			{

				// OPTIMIZATION: for UINT32 compares avoid func call by doing
				// compare here!

				if (pLhs->eType == FLM_UINT32_VAL && pRhs->eType == FLM_UINT32_VAL)
				{
					*puiTrueFalse =
						(FQ_COMPARE( pLhs->val.uiVal, pRhs->val.uiVal) == 0)
							? FLM_TRUE : FLM_FALSE;
				}
				else
				{
					*puiTrueFalse =
						(flmCurDoRelationalOp( pLhs, pRhs, uiLang) == 0)
							? FLM_TRUE : FLM_FALSE;
				}
				
				break;
			}
			
			case FLM_MATCH_OP:
			{
				if ((pLhs->uiFlags & FLM_WILD) || (pRhs->uiFlags & FLM_WILD))
				{
					*puiTrueFalse = flmCurDoMatchOp( pLhs, pRhs, uiLang, FALSE, FALSE);
				}
				else
				{
					*puiTrueFalse =
						(flmCurDoRelationalOp( pLhs, pRhs, uiLang) == 0)
							? FLM_TRUE : FLM_FALSE;
				}
				
				break;
			}
			
			case FLM_MATCH_BEGIN_OP:
			{
				*puiTrueFalse = flmCurDoMatchOp( pLhs, pRhs, uiLang, FALSE, TRUE);
				break;
			}
			
			case FLM_MATCH_END_OP:
			{
				*puiTrueFalse = flmCurDoMatchOp( pLhs, pRhs, uiLang, TRUE, FALSE);
				break;
			}
			
			case FLM_NE_OP:
			{
				*puiTrueFalse =
					(flmCurDoRelationalOp( pLhs, pRhs, uiLang) != 0)
						? FLM_TRUE : FLM_FALSE;
				break;
			}
			
			case FLM_LT_OP:
			{
				*puiTrueFalse =
					(flmCurDoRelationalOp( pLhs, pRhs, uiLang) < 0)
						? FLM_TRUE : FLM_FALSE;
				break;
			}
			
			case FLM_LE_OP:
			{
				*puiTrueFalse =
					(flmCurDoRelationalOp( pLhs, pRhs, uiLang) <= 0)
						? FLM_TRUE : FLM_FALSE;
				break;
			}
			
			case FLM_GT_OP:
			{
				*puiTrueFalse =
					(flmCurDoRelationalOp( pLhs, pRhs, uiLang) > 0)
						? FLM_TRUE : FLM_FALSE;
				break;
			}
			
			case FLM_GE_OP:
			{
				*puiTrueFalse =
					(flmCurDoRelationalOp( pLhs, pRhs, uiLang) >= 0)
						? FLM_TRUE : FLM_FALSE;
				break;
			}
			
			case FLM_CONTAINS_OP:
			{
				*puiTrueFalse = flmCurDoContainsOp( pLhs, pRhs, uiLang);
				break;
			}
			
			default:
			{
				// Syntax error.
				
				*puiTrueFalse = 0;
				flmAssert( 0);
				break;
			}
		}
	}
}

/****************************************************************************
Desc: Performs relational operations on stack elements.
****************************************************************************/
RCODE flmCurEvalCompareOp(
	FDB *				pDb,
	SUBQUERY *		pSubQuery,
	FlmRecord *		pRecord,
	FQNODE *			pQNode,
	QTYPES			eOp,
	FLMBOOL			bHaveKey,
	FQATOM *			pResult)
{
	RCODE				rc = FERR_OK;
	FQNODE *			pTmpQNode;
	FQATOM *			pTmpQAtom;
	FQATOM *			pLhs;
	FQATOM *			pRhs;
	FQATOM *			pFirstRhs;
	FQATOM			Lhs;
	FQATOM			Rhs;
	QTYPES			wTmpOp = eOp;
	QTYPES			eType;
	QTYPES			eFldType = NO_TYPE;
	FLMUINT			uiTrueFalse = 0;
	FLMBOOL			bSecondOperand;
	FLMBOOL			bSwitchOperands = FALSE;
	FLMBOOL			bGetNewField = FALSE;
	FLMBOOL			bRightTruncated = FALSE;
	FLMBOOL			bNotted = (pQNode->uiStatus & FLM_NOTTED) ? TRUE : FALSE;
	FLMBOOL			bResolveUnknown = (pQNode->uiStatus & FLM_RESOLVE_UNK) ? TRUE : FALSE;
	FLMBOOL			bForEvery = (pQNode->uiStatus & FLM_FOR_EVERY) ? TRUE : FALSE;
	FQNODE *			pRightOpCB = NULL;
	FQNODE *			pLeftOpCB = NULL;
	FQNODE *			pOpCB = NULL;
	RCODE				TempRc;
	FLMBOOL			bSavedInvisTrans;
	POOL *			pTmpPool = &pDb->TempPool;
	void *			pvMark = GedPoolMark( pTmpPool);

	pResult->eType = FLM_BOOL_VAL;
	pResult->pNext = NULL;
	pResult->val.uiBool = 0;
	if (pQNode->pChild == NULL)
	{
		rc = RC_SET( FERR_CURSOR_SYNTAX);
		goto Exit;
	}

	pLhs = &Lhs;
	pRhs = &Rhs;

	pTmpQNode = pQNode->pChild;
	bSecondOperand = FALSE;
	f_memset( &Lhs, 0, sizeof(FQATOM));
	f_memset( &Rhs, 0, sizeof(FQATOM));
	pLhs->eType = pRhs->eType = NO_TYPE;

	// Get the two operands from the stack or passed-in record node

	pTmpQAtom = pLhs;
	
Get_Operand:

	eType = GET_QNODE_TYPE( pTmpQNode);
	if (IS_FLD_CB( eType, pTmpQNode))
	{
		eType = FLM_CB_FLD;
	}

	if (IS_VAL( eType))
	{
		if (bSecondOperand)
		{
			pRhs = pTmpQNode->pQAtom;
		}
		else
		{
			pLhs = pTmpQNode->pQAtom;
			bSwitchOperands = TRUE;
		}
	}
	else if (eType == FLM_FLD_PATH || eType == FLM_CB_FLD)
	{
		if (bSecondOperand)
		{
			eFldType = pLhs->eType;
			if (eType == FLM_CB_FLD)
			{
				pOpCB = pRightOpCB = pTmpQNode;
			}
		}
		else
		{
			if (pTmpQNode->pNextSib == NULL)
			{
				rc = RC_SET( FERR_CURSOR_SYNTAX);
				goto Exit;
			}

			eFldType = GET_QNODE_TYPE( pTmpQNode->pNextSib);
			if (eType == FLM_CB_FLD)
			{
				pOpCB = pLeftOpCB = pTmpQNode;
			}
		}

		if (!IS_VAL( eFldType))
		{
			eFldType = NO_TYPE;
		}

		if (eType == FLM_CB_FLD)
		{

			// Get the first occurrence of the field.

			if (RC_BAD( rc = flmFieldIterate( pDb, pTmpPool, eFldType, pOpCB,
						  pRecord, bHaveKey, TRUE, FLM_FLD_FIRST, pTmpQAtom)))
			{
				goto Exit;
			}

			if (pTmpQAtom->uiFlags & FLM_IS_RIGHT_TRUNCATED_DATA)
			{
				bRightTruncated = TRUE;
			}
		}
		else
		{
			if (RC_BAD( rc = flmCurGetAtomFromRec( pDb, pTmpPool,
						  pTmpQNode->pQAtom, pRecord, eFldType, TRUE, pTmpQAtom,
						  bHaveKey)))
			{
				goto Exit;
			}

			if (pTmpQAtom->uiFlags & FLM_IS_RIGHT_TRUNCATED_DATA)
			{
				bRightTruncated = TRUE;
			}
		}

		// Check to see if this field is a substring field in the index. If
		// it is, and it is not the first substring value in the field, and
		// we are doing a match begin or match operator, return FLM_FALSE -
		// we cannot evaluate anything except first substrings in these two
		// cases. NOTE: If we are evaluating a key and this is a callback
		// field, we don't need to worry about this condition, because the
		// CB field will have been set up to return unknown.

		if (bHaveKey &&
			 (pTmpQAtom->uiFlags & FLM_IS_LEFT_TRUNCATED_DATA) &&
			 (eOp == FLM_MATCH_OP || eOp == FLM_MATCH_BEGIN_OP))
		{
			pResult->val.uiBool = FLM_FALSE;
			goto Exit;
		}
	}
	else if (IS_ARITH_OP( eType))
	{
		if (RC_BAD( rc = flmCurEvalArithOp( pDb, pSubQuery, pRecord, pTmpQNode,
					  eType, bGetNewField, bHaveKey, pTmpQAtom)))
		{
			goto Exit;
		}
	}
	else
	{
		rc = RC_SET( FERR_CURSOR_SYNTAX);
		goto Exit;
	}

	if (!bSecondOperand)
	{
		if (pTmpQNode->pNextSib == NULL)
		{
			rc = RC_SET( FERR_CURSOR_SYNTAX);
			goto Exit;
		}

		pTmpQNode = pTmpQNode->pNextSib;
		pTmpQAtom = pRhs;
		bSecondOperand = TRUE;
		goto Get_Operand;
	}

	// If necessary, reverse the operator to render the expression in the
	// form <field><op><value>.

	if (bSwitchOperands)
	{
		if (REVERSIBLE( eOp))
		{
			wTmpOp = DO_REVERSE( eOp);
			pLhs = &Rhs;
			pRhs = &Lhs;
		}
		else
		{
			rc = RC_SET( FERR_CURSOR_SYNTAX);
			goto Exit;
		}
	}

	// Now do the operation using our operators.

	pFirstRhs = pRhs;
	for (;;)
	{
		FLMBOOL	bDoComp = TRUE;

		// If this key piece is truncated, and the selection criteria can't
		// be evaluated as a result, read the record and start again. NOTE:
		// this will only happen if the field type is text or binary.

		if (bHaveKey &&
			 bRightTruncated &&
			 pRhs->eType != FLM_UNKNOWN &&
			 pLhs->eType != FLM_UNKNOWN)
		{

			// VISIT: We should optimized to flunk or pass text compares.
			// The problems come with comparing only up to the first
			// wildcard.

			if (pLhs->eType != FLM_BINARY_VAL)
			{
				uiTrueFalse = FLM_UNK;
			}
			else
			{
				FLMINT	iCompVal;

				// We better only compare binary types here.

				flmAssert( pRhs->eType == FLM_BINARY_VAL);

				iCompVal = f_memcmp( pLhs->val.pucBuf, pRhs->val.pucBuf,
										  f_min( pLhs->uiBufLen, pRhs->uiBufLen));

				if (!iCompVal)
				{

					// Lhs is the truncated key. If its length is <= to the
					// length of the Rhs, comparison must continue by fetching
					// the record. So, we set uiTrueFalse to FLM_UNK.
					// Otherwise, we know that the Lhs length is greater than
					// the Rhs, so we are able to complete the comparison even
					// though the key is truncated.

					if (pLhs->uiBufLen <= pRhs->uiBufLen)
					{
						uiTrueFalse = FLM_UNK;
					}
					else
					{
						iCompVal = 1;
					}
				}

				// iCompVal == 0 has been handled above. This means that
				// uiTrueFalse has been set to FLM_UNK.

				if (iCompVal)
				{
					switch (eOp)
					{
						case FLM_NE_OP:
						{

							// We know that iCompVal != 0

							uiTrueFalse = FLM_TRUE;
							break;
						}
						
						case FLM_GT_OP:
						case FLM_GE_OP:
						{
							uiTrueFalse = (iCompVal > 0) ? FLM_TRUE : FLM_FALSE;
							break;
						}
						
						case FLM_LT_OP:
						case FLM_LE_OP:
						{
							uiTrueFalse = (iCompVal < 0) ? FLM_TRUE : FLM_FALSE;
							break;
						}
						
						case FLM_EQ_OP:
						default:
						{
							// We know that iCompVal != 0

							uiTrueFalse = FLM_FALSE;
							break;
						}
					}
				}

				bDoComp = FALSE;
			}
		}
		else
		{
			flmCompareOperands( pSubQuery->uiLanguage, pLhs, pRhs, eOp,
									 bResolveUnknown, bForEvery, bNotted, bHaveKey,
									 &uiTrueFalse);
		}

		if (bNotted)
		{
			uiTrueFalse = (uiTrueFalse == FLM_TRUE) 
									? FLM_FALSE 
									: (uiTrueFalse == FLM_FALSE) 
											? FLM_TRUE 
											: FLM_UNK;
		}

		// For index keys - validate that the field is correct if the
		// compare returned true. Otherwise, set the result to unknown.
		// VISIT: This will not work for index keys that have more than one
		// field that needs to be validated.

		if (bDoComp && eType == FLM_FLD_PATH &&
			 uiTrueFalse == FLM_TRUE && bHaveKey)
		{
			FQATOM *		pTreeAtom = pTmpQNode->pQAtom;
			FLMUINT		uiResult;
			void *		pField = NULL;

			CB_ENTER( pDb, &bSavedInvisTrans);
			rc = pTreeAtom->val.QueryFld.fnGetField( 
				pTreeAtom->val.QueryFld.pvUserData, pRecord, (HFDB) pDb,
				pTreeAtom->val.QueryFld.puiFldPath, FLM_FLD_VALIDATE, NULL,
				&pField, &uiResult);
			CB_EXIT( pDb, bSavedInvisTrans);

			if (RC_BAD( rc))
			{
				goto Exit;
			}
			else if (uiResult == FLM_UNK)
			{
				uiTrueFalse = FLM_UNK;
			}
			else if (uiResult == FLM_FALSE)
			{
				uiTrueFalse = FLM_FALSE;
			}
		}

		pResult->val.uiBool = uiTrueFalse;

		// Doing contextless, see if we need to process any more. If the
		// FOR EVERY flag is TRUE (universal quantifier), we quit when we
		// see a FALSE. If the FOR EVERY flag is FALSE (existential
		// quantifier), we quit when we see a TRUE.

		if ((bForEvery && uiTrueFalse == FLM_FALSE) ||
			 (!bForEvery && uiTrueFalse == FLM_TRUE))
		{
			break;
		}

		// Get the next right hand operand.

		if (!pRightOpCB)
		{
			pRhs = pRhs->pNext;
		}
		else if (pRhs->eType == FLM_UNKNOWN)
		{
			pRhs = NULL;
		}
		else
		{
			if (RC_BAD( rc = flmFieldIterate( pDb, pTmpPool, eFldType, pRightOpCB,
						  pRecord, bHaveKey, TRUE, FLM_FLD_NEXT, pRhs)))
			{
				goto Exit;
			}

			if (pRhs->uiFlags & FLM_IS_RIGHT_TRUNCATED_DATA)
			{
				bRightTruncated = TRUE;
			}

			if (pRhs->eType == FLM_UNKNOWN)
			{
				pRhs = NULL;
			}
		}

		// If no more right hand side, get the next left hand side, and
		// reset the right hand side.

		if (!pRhs)
		{
			if (!pLeftOpCB)
			{
				pLhs = pLhs->pNext;
			}
			else if (pLhs->eType == FLM_UNKNOWN)
			{
				pLhs = NULL;
			}
			else
			{
				if (RC_BAD( rc = flmFieldIterate( pDb, pTmpPool, eFldType,
							  pLeftOpCB, pRecord, bHaveKey, TRUE, FLM_FLD_NEXT, pLhs)))
				{
					goto Exit;
				}

				if (pLhs->uiFlags & FLM_IS_RIGHT_TRUNCATED_DATA)
				{
					bRightTruncated = TRUE;
				}

				if (pLhs->eType == FLM_UNKNOWN)
				{
					pLhs = NULL;
				}
			}

			if (!pLhs)
			{
				break;
			}

			// Reset the right hand side to the first.

			if (pRightOpCB)
			{
				if (RC_BAD( rc = flmFieldIterate( pDb, pTmpPool, eFldType,
					pRightOpCB, pRecord, bHaveKey, TRUE, FLM_FLD_FIRST, pRhs)))
				{
					goto Exit;
				}

				if (pRhs->uiFlags & FLM_IS_RIGHT_TRUNCATED_DATA)
				{
					bRightTruncated = TRUE;
				}
			}
			else
			{
				pRhs = pFirstRhs;
			}
		}
	}

Exit:

	// Clean up any field callbacks.

	if (pLeftOpCB)
	{
		CB_ENTER( pDb, &bSavedInvisTrans);
		TempRc = pLeftOpCB->pQAtom->val.QueryFld.fnGetField( 
			pLeftOpCB->pQAtom->val.QueryFld.pvUserData, NULL, (HFDB) pDb,
			pLeftOpCB->pQAtom->val.QueryFld.puiFldPath, FLM_FLD_RESET, NULL,
			NULL, NULL);
		CB_EXIT( pDb, bSavedInvisTrans);

		if (RC_BAD( TempRc))
		{
			if (RC_OK( rc))
			{
				rc = TempRc;
			}
		}
	}

	if (pRightOpCB)
	{
		CB_ENTER( pDb, &bSavedInvisTrans);
		TempRc = pRightOpCB->pQAtom->val.QueryFld.fnGetField( 
			pRightOpCB->pQAtom->val.QueryFld.pvUserData, NULL, (HFDB) pDb,
			pRightOpCB->pQAtom->val.QueryFld.puiFldPath, FLM_FLD_RESET, NULL,
			NULL, NULL);
		CB_EXIT( pDb, bSavedInvisTrans);
		if (RC_BAD( TempRc))
		{
			if (RC_OK( rc))
			{
				rc = TempRc;
			}
		}
	}

	GedPoolReset( pTmpPool, pvMark);
	return (rc);
}

/****************************************************************************
Desc: Performs logical AND or OR operations
****************************************************************************/
FSTATIC RCODE flmCurEvalLogicalOp(
	FDB *			pDb,
	SUBQUERY *	pSubQuery,
	FlmRecord *	pRecord,
	FQNODE *		pQNode,
	QTYPES		eOp,
	FLMBOOL		bHaveKey,
	FQATOM *		pResult)
{
	RCODE			rc = FERR_OK;
	FQATOM		TmpQAtom;
	FQNODE *		pTmpQNode;
	FQATOM *		pTmpQAtom;
	QTYPES		eType;
	FLMBOOL		bSavedInvisTrans;
	FLMUINT		uiTrueFalse;
	RCODE			TempRc;

	pResult->eType = FLM_BOOL_VAL;
	pResult->pNext = NULL;
	pResult->val.uiBool = 0;

	if (pQNode->pChild == NULL)
	{
		rc = RC_SET( FERR_CURSOR_SYNTAX);
		goto Exit;
	}

	FLM_SET_RESULT( pQNode->uiStatus, 0);
	pTmpQNode = pQNode->pChild;

Get_Operand:

	// Get the operand to process

	pTmpQAtom = &TmpQAtom;
	pTmpQAtom->pNext = NULL;
	pTmpQAtom->pFieldRec = NULL;
	pTmpQAtom->eType = NO_TYPE;
	pTmpQAtom->uiBufLen = 0;
	pTmpQAtom->val.uiVal = 0;

	eType = GET_QNODE_TYPE( pTmpQNode);
	if (IS_FLD_CB( eType, pTmpQNode))
	{
		eType = FLM_CB_FLD;
	}

	if (IS_VAL( eType))
	{
		pTmpQAtom = pTmpQNode->pQAtom;
	}
	else if (eType == FLM_CB_FLD)
	{

		// Get the first occurrence of the field.

		if (RC_OK( rc = flmFieldIterate( pDb, &pDb->TempPool, NO_TYPE, pTmpQNode,
					 pRecord, bHaveKey, FALSE, FLM_FLD_FIRST, pTmpQAtom)))
		{
			if (pTmpQNode->uiStatus & FLM_NOTTED &&
				 pTmpQAtom->eType == FLM_BOOL_VAL)
			{
				pTmpQAtom->val.uiBool = (pTmpQAtom->val.uiBool == FLM_TRUE) 
														? FLM_FALSE 
														: FLM_TRUE;
			}
		}

		CB_ENTER( pDb, &bSavedInvisTrans);
		TempRc = pTmpQNode->pQAtom->val.QueryFld.fnGetField( 
			pTmpQNode->pQAtom->val.QueryFld.pvUserData, NULL, (HFDB) pDb,
			pTmpQNode->pQAtom->val.QueryFld.puiFldPath, FLM_FLD_RESET, NULL,
			NULL, NULL);
		CB_EXIT( pDb, bSavedInvisTrans);

		if (RC_BAD( TempRc) && RC_OK( rc))
		{
			rc = TempRc;
		}

		if (RC_BAD( rc))
		{
			goto Exit;
		}
	}
	else if (eType == FLM_FLD_PATH)
	{
		if (RC_BAD( rc = flmCurGetAtomFromRec( pDb, &pDb->TempPool,
					  pTmpQNode->pQAtom, pRecord, NO_TYPE, FALSE, pTmpQAtom, bHaveKey
					  )))
		{
			goto Exit;
		}

		// NOTE: pTmpQAtom could come back from this as an UNKNOWN now,
		// even though we are testing for field existence. This could happen
		// when we are testing a key and we have a callback, but the
		// callback cannot tell if the field instance is actually present or
		// not.

		if ((pTmpQNode->uiStatus & FLM_NOTTED) &&
			 (pTmpQAtom->eType == FLM_BOOL_VAL))
		{
			pTmpQAtom->val.uiBool = (pTmpQAtom->val.uiBool == FLM_TRUE) 
														? FLM_FALSE 
														: FLM_TRUE;
		}
	}
	else if (IS_LOG_OP( eType))
	{

		// Traverse down the tree.

		pQNode = pTmpQNode;
		eOp = eType;
		FLM_SET_RESULT( pQNode->uiStatus, 0);
		pTmpQNode = pTmpQNode->pChild;
		goto Get_Operand;
	}
	else if (IS_COMPARE_OP( eType))
	{
		if (RC_BAD( rc = flmCurEvalCompareOp( pDb, pSubQuery, pRecord, pTmpQNode,
					  eType, bHaveKey, pTmpQAtom)))
		{
			goto Exit;
		}
	}
	else if (eType == FLM_USER_PREDICATE)
	{
		if (bHaveKey)
		{

			// Don't want to do the callback if we only have a key - because
			// the callback won't have access to all of the values from here.
			// The safe thing is to just return unknown.

			pResult->eType = FLM_UNKNOWN;
			goto Exit;
		}
		else
		{
			CB_ENTER( pDb, &bSavedInvisTrans);
			rc = pTmpQNode->pQAtom->val.pPredicate->testRecord( 
					(HFDB) pDb, pRecord, pRecord->getID(), &pTmpQAtom->val.uiBool);
			CB_EXIT( pDb, bSavedInvisTrans);
			if (RC_BAD( rc))
			{
				goto Exit;
			}

			pTmpQAtom->eType = FLM_BOOL_VAL;
		}
	}
	else
	{
		rc = RC_SET( FERR_CURSOR_SYNTAX);
		goto Exit;
	}

	// See what our TRUE/FALSE result is.

	uiTrueFalse = flmCurEvalTrueFalse( pTmpQAtom);

	// Traverse back up the tree, ORing or ANDing or NOTing this result as
	// necessary.

	for (;;)
	{

		// If ANDing and we have a FALSE result or ORing and we have a TRUE
		// result, the result can simply be propagated up the tree.

		if ((eOp == FLM_AND_OP && uiTrueFalse == FLM_FALSE) ||
			 (eOp == FLM_OR_OP && uiTrueFalse == FLM_TRUE))
		{

			// We are done if we can go no higher in the tree.

			pTmpQNode = pQNode;
			if ((pQNode = pQNode->pParent) == NULL)
			{
				break;
			}

			eOp = GET_QNODE_TYPE( pQNode);
		}
		else if (pTmpQNode->pNextSib)
		{

			// Can only be one operand of a NOT operator.

			flmAssert( eOp != FLM_NOT_OP);

			// Save the left-hand side result into pQNode->uiStatus

			FLM_SET_RESULT( pQNode->uiStatus, uiTrueFalse);
			pTmpQNode = pTmpQNode->pNextSib;
			goto Get_Operand;
		}
		else		// Processing results of right hand operand
		{
			FLMUINT	uiRhs;

			if (eOp == FLM_AND_OP)
			{

				// FALSE case for AND operator has already been handled up
				// above.

				flmAssert( uiTrueFalse != FLM_FALSE);

				// AND the results from the left-hand side. Get left-hand
				// side result from pQNode.

				uiRhs = uiTrueFalse;
				uiTrueFalse = FLM_GET_RESULT( pQNode->uiStatus);

				// Perform logical AND operation.

				if (uiRhs & FLM_FALSE)
				{
					uiTrueFalse |= FLM_FALSE;
				}

				if (uiRhs & FLM_UNK)
				{
					uiTrueFalse |= FLM_UNK;
				}

				// If both left hand side and right hand side do not have
				// FLM_TRUE set, we must turn it off.

				if ((uiTrueFalse & FLM_TRUE) && (!(uiRhs & FLM_TRUE)))
				{
					uiTrueFalse &= (~(FLM_TRUE));
				}
			}
			else if (eOp == FLM_OR_OP)
			{

				// TRUE case for OR operator better have been handled up
				// above.

				flmAssert( uiTrueFalse != FLM_TRUE);

				// OR the results from the left hand side. Get left-hand side
				// result from pQNode.

				uiRhs = uiTrueFalse;
				uiTrueFalse = FLM_GET_RESULT( pQNode->uiStatus);

				// Perform logical OR operation.

				if (uiRhs & FLM_TRUE)
				{
					uiTrueFalse |= FLM_TRUE;
				}

				if (uiRhs & FLM_UNK)
				{
					uiTrueFalse |= FLM_UNK;
				}

				// If both left hand side and right hand side do not have
				// FLM_FALSE set, we must turn it off.

				if ((uiTrueFalse & FLM_FALSE) && (!(uiRhs & FLM_FALSE)))
				{
					uiTrueFalse &= (~(FLM_FALSE));
				}
			}
			else	// (eOp == FLM_NOT_OP)
			{
				flmAssert( eOp == FLM_NOT_OP);

				// NOT the result

				if (uiTrueFalse == FLM_TRUE)
				{
					uiTrueFalse = FLM_FALSE;
				}
				else if (uiTrueFalse == FLM_FALSE)
				{
					uiTrueFalse = FLM_TRUE;
				}
				else if (uiTrueFalse == (FLM_UNK | FLM_TRUE))
				{
					uiTrueFalse = FLM_FALSE | FLM_UNK;
				}
				else if (uiTrueFalse == (FLM_UNK | FLM_FALSE))
				{
					uiTrueFalse = FLM_TRUE | FLM_UNK;
				}
			}

			// Traverse back up to the parent with this result.

			pTmpQNode = pQNode;

			// We are done if we are at the top of the tree.

			if ((pQNode = pQNode->pParent) == NULL)
			{
				break;
			}

			eOp = GET_QNODE_TYPE( pQNode);
		}
	}

	// At this point, we are done, because there is no higher to traverse
	// back up in the tree.

	pResult->val.uiBool = uiTrueFalse;

Exit:

	return (rc);
}

/****************************************************************************
Desc:	Checks a record that has been retrieved from the database
		to see if it matches the criteria specified in the query
		stack.
****************************************************************************/
RCODE flmCurEvalCriteria(
	CURSOR *			pCursor,
	SUBQUERY *		pSubQuery,
	FlmRecord *		pRecord,
	FLMBOOL			bHaveKey,
	FLMUINT *		puiResult)
{
	RCODE				rc = FERR_OK;
	FQATOM			Result;
	QTYPES			eType;
	FDB *				pDb = pCursor->pDb;
	FQNODE *			pQNode;
	void *			pTmpMark = GedPoolMark( &pDb->TempPool);
	FLMUINT			uiResult = 0;
	FQNODE *			pOpCB = NULL;
	RCODE				TempRc;

	// By definition, a NULL record doesn't match selection criteria.

	if (!pRecord)
	{
		uiResult = FLM_FALSE;
		goto Exit;
	}

	// Record's container ID must match the cursor's

	if (pRecord->getContainerID() != pCursor->uiContainer)
	{
		uiResult = FLM_FALSE;
		goto Exit;
	}

	if (!pSubQuery->pTree)
	{
		uiResult = FLM_TRUE;
		goto Exit;
	}

	// First check the record type if necessary, then verify that there
	// are search criteria to match against.

	if (pCursor->uiRecType)
	{
		void *	pField = pRecord->root();

		if (!pField || pCursor->uiRecType != pRecord->getFieldID( pField))
		{
			uiResult = FLM_FALSE;
			goto Exit;
		}
	}

	pQNode = pSubQuery->pTree;

	f_memset( &Result, 0, sizeof(FQATOM));

	eType = GET_QNODE_TYPE( pQNode);
	if (IS_FLD_CB( eType, pQNode))
	{
		eType = FLM_CB_FLD;
	}

	if (IS_VAL( eType))
	{
		uiResult = flmCurEvalTrueFalse( pQNode->pQAtom);
	}
	else if (eType == FLM_USER_PREDICATE)
	{
		if (bHaveKey)
		{

			// Don't want to do the callback if we only have a key - because
			// the callback won't have access to all of the values from here.
			// The safe thing is to just return unknown.

			uiResult = FLM_UNK;
			rc = FERR_OK;
		}
		else
		{
			FLMBOOL	bSavedInvisTrans;
			CB_ENTER( pDb, &bSavedInvisTrans);
			rc = pQNode->pQAtom->val.pPredicate->testRecord( 
				(HFDB) pDb, pRecord, pRecord->getID(), &uiResult);
			CB_EXIT( pDb, bSavedInvisTrans);
			if (RC_BAD( rc))
			{
				goto Exit;
			}
		}
	}
	else
	{
		if (eType == FLM_CB_FLD)
		{

			// Get the first occurrence of the field.

			pOpCB = pQNode;
			if (RC_BAD( rc = flmFieldIterate( pDb, &pDb->TempPool, NO_TYPE, pQNode,
						  pRecord, bHaveKey, FALSE, FLM_FLD_FIRST, &Result)))
			{
				goto Exit;
			}

			if (pQNode->uiStatus & FLM_NOTTED && Result.eType == FLM_BOOL_VAL)
			{
				Result.val.uiBool = (Result.val.uiBool == FLM_TRUE) 
													? FLM_FALSE 
													: FLM_TRUE;
			}
		}
		else if (eType == FLM_FLD_PATH)
		{
			if (RC_BAD( rc = flmCurGetAtomFromRec( pDb, &pDb->TempPool,
						  	pQNode->pQAtom, pRecord, NO_TYPE, FALSE, &Result, 
							bHaveKey)))
			{
				goto Exit;
			}

			// NOTE: Result could come back from this as an UNKNOWN now,
			// even though we are testing for field existence. This could
			// happen when we are testing a key and we have a callback, but
			// the callback cannot tell if the field instance is actually
			// present or not.

			if ((pQNode->uiStatus & FLM_NOTTED) && (Result.eType == FLM_BOOL_VAL))
			{
				Result.val.uiBool = (Result.val.uiBool == FLM_TRUE) 
													? FLM_FALSE 
													: FLM_TRUE;
			}
		}
		else if (IS_LOG_OP( eType))
		{
			if (RC_BAD( rc = flmCurEvalLogicalOp( pDb, pSubQuery, pRecord, pQNode,
						  eType, bHaveKey, &Result)))
			{
				goto Exit;
			}
		}
		else if (IS_COMPARE_OP( eType))
		{
			if (RC_BAD( rc = flmCurEvalCompareOp( pDb, pSubQuery, pRecord, pQNode,
						  eType, bHaveKey, &Result)))
			{
				goto Exit;
			}
		}
		else
		{
			uiResult = FLM_FALSE;
			rc = RC_SET( FERR_CURSOR_SYNTAX);
			goto Exit;
		}

		uiResult = flmCurEvalTrueFalse( &Result);

		if (!bHaveKey && uiResult == FLM_UNK)
		{
			uiResult = FLM_FALSE;
		}
	}

Exit:

	if (rc == FERR_EOF_HIT)
	{
		rc = FERR_OK;
	}

	// Clean up any field callbacks.

	if (pOpCB)
	{
		FLMBOOL	bSavedInvisTrans;
		CB_ENTER( pDb, &bSavedInvisTrans);
		TempRc = pOpCB->pQAtom->val.QueryFld.fnGetField( 
			pOpCB->pQAtom->val.QueryFld.pvUserData, NULL, (HFDB) pDb,
			pOpCB->pQAtom->val.QueryFld.puiFldPath, FLM_FLD_RESET, NULL, NULL,
			NULL);
		CB_EXIT( pDb, bSavedInvisTrans);
		if (RC_BAD( TempRc))
		{
			if (RC_OK( rc))
			{
				rc = TempRc;
			}
		}
	}

	GedPoolReset( &pDb->TempPool, pTmpMark);
	*puiResult = uiResult;
	return (rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC RCODE OpSyntaxError(
	FQATOM *		pLhs,
	FQATOM *		pRhs,
	FQATOM *		pResult)
{
	F_UNREFERENCED_PARM( pLhs);
	F_UNREFERENCED_PARM( pRhs);
	F_UNREFERENCED_PARM( pResult);
	
	return (RC_SET( FERR_CURSOR_SYNTAX));
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC RCODE OpUUBitAND(
	FQATOM *	pLhs,
	FQATOM *	pRhs,
	FQATOM *	pResult)
{
	pResult->val.uiVal = pLhs->val.uiVal & pRhs->val.uiVal;
	pResult->eType = FLM_UINT32_VAL;
	return (FERR_OK);
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC RCODE OpUUBitOR(
	FQATOM *	pLhs,
	FQATOM *	pRhs,
	FQATOM *	pResult)
{
	pResult->val.uiVal = pLhs->val.uiVal | pRhs->val.uiVal;
	pResult->eType = FLM_UINT32_VAL;
	return (FERR_OK);
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC RCODE OpUUBitXOR(
	FQATOM *	pLhs,
	FQATOM *	pRhs,
	FQATOM *	pResult)
{
	pResult->val.uiVal = pLhs->val.uiVal ^ pRhs->val.uiVal;
	pResult->eType = FLM_UINT32_VAL;
	return (FERR_OK);
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC RCODE OpUUMult(
	FQATOM *	pLhs,
	FQATOM *	pRhs,
	FQATOM *	pResult)
{
	pResult->val.uiVal = pLhs->val.uiVal * pRhs->val.uiVal;
	pResult->eType = FLM_UINT32_VAL;
	return (FERR_OK);
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC RCODE OpUSMult(
	FQATOM *	pLhs,
	FQATOM *	pRhs,
	FQATOM *	pResult)
{
	pResult->val.iVal = (FLMINT) pLhs->val.uiVal * pRhs->val.iVal;
	pResult->eType = FLM_INT32_VAL;
	return (FERR_OK);
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC RCODE OpSSMult(
	FQATOM *	pLhs,
	FQATOM *	pRhs,
	FQATOM *	pResult)
{
	pResult->val.iVal = pLhs->val.iVal * pRhs->val.iVal;
	pResult->eType = (pResult->val.iVal < 0) ? FLM_INT32_VAL : FLM_UINT32_VAL;
	return (FERR_OK);
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC RCODE OpSUMult(
	FQATOM *	pLhs,
	FQATOM *	pRhs,
	FQATOM *	pResult)
{
	pResult->val.iVal = pLhs->val.iVal * (FLMINT) pRhs->val.uiVal;
	pResult->eType = FLM_INT32_VAL;
	return (FERR_OK);
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC RCODE OpURMult(
	FQATOM *	pLhs,
	FQATOM *	pRhs,
	FQATOM *	pResult)
{
	return (OpSyntaxError( pLhs, pRhs, pResult));
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC RCODE OpRUMult(
	FQATOM *	pLhs,
	FQATOM *	pRhs,
	FQATOM *	pResult)
{
	return (OpSyntaxError( pLhs, pRhs, pResult));
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC RCODE OpSRMult(
	FQATOM *	pLhs,
	FQATOM *	pRhs,
	FQATOM *	pResult)
{
	return (OpSyntaxError( pLhs, pRhs, pResult));
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC RCODE OpRSMult(
	FQATOM *	pLhs,
	FQATOM *	pRhs,
	FQATOM *	pResult)
{
	return (OpSyntaxError( pLhs, pRhs, pResult));
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC RCODE OpRRMult(
	FQATOM *	pLhs,
	FQATOM *	pRhs,
	FQATOM *	pResult)
{
	return (OpSyntaxError( pLhs, pRhs, pResult));
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC RCODE OpUUDiv(
	FQATOM *	pLhs,
	FQATOM *	pRhs,
	FQATOM *	pResult)
{
	if (pRhs->val.uiVal)
	{
		pResult->val.uiVal = pLhs->val.uiVal / pRhs->val.uiVal;
		pResult->eType = FLM_UINT32_VAL;
	}
	else
	{
		// Divide by ZERO case.
		
		pResult->val.uiVal = 0;
		pResult->eType = FLM_UNKNOWN;
	}

	return (FERR_OK);
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC RCODE OpUSDiv(
	FQATOM *	pLhs,
	FQATOM *	pRhs,
	FQATOM *	pResult)
{
	if (pRhs->val.iVal)
	{
		pResult->val.iVal = (FLMINT) pLhs->val.uiVal / pRhs->val.iVal;
		pResult->eType = FLM_INT32_VAL;
	}
	else
	{
		// Divide by ZERO case.

		pResult->val.uiVal = 0;
		pResult->eType = FLM_UNKNOWN;
	}

	return (FERR_OK);
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC RCODE OpSSDiv(
	FQATOM *	pLhs,
	FQATOM *	pRhs,
	FQATOM *	pResult)
{
	if (pRhs->val.iVal)
	{
		pResult->val.iVal = pLhs->val.iVal / pRhs->val.iVal;
		pResult->eType = (pResult->val.iVal < 0) ? FLM_INT32_VAL : FLM_UINT32_VAL;
	}
	else
	{
		pResult->val.uiVal = 0; // divide by ZERO case
		pResult->eType = FLM_UNKNOWN;
	}

	return (FERR_OK);
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC RCODE OpSUDiv(
	FQATOM *	pLhs,
	FQATOM *	pRhs,
	FQATOM *	pResult)
{
	if (pRhs->val.uiVal)
	{
		pResult->val.iVal = pLhs->val.iVal / (FLMINT) pRhs->val.uiVal;
		pResult->eType = FLM_INT32_VAL;
	}
	else	// Divide by ZERO case - let's try not to crash.
	{
		pResult->val.uiVal = 0;
		pResult->eType = FLM_UNKNOWN;
	}

	return (FERR_OK);
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC RCODE OpURDiv(
	FQATOM *	pLhs,
	FQATOM *	pRhs,
	FQATOM *	pResult)
{
	return (OpSyntaxError( pLhs, pRhs, pResult));
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC RCODE OpRUDiv(
	FQATOM *	pLhs,
	FQATOM *	pRhs,
	FQATOM *	pResult)
{
	return (OpSyntaxError( pLhs, pRhs, pResult));
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC RCODE OpSRDiv(
	FQATOM *	pLhs,
	FQATOM *	pRhs,
	FQATOM *	pResult)
{
	return (OpSyntaxError( pLhs, pRhs, pResult));
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC RCODE OpRSDiv(
	FQATOM *	pLhs,
	FQATOM *	pRhs,
	FQATOM *	pResult)
{
	return (OpSyntaxError( pLhs, pRhs, pResult));
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC RCODE OpRRDiv(
	FQATOM *	pLhs,
	FQATOM *	pRhs,
	FQATOM *	pResult)
{
	return (OpSyntaxError( pLhs, pRhs, pResult));
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC RCODE OpUUMod(
	FQATOM *	pLhs,
	FQATOM *	pRhs,
	FQATOM *	pResult)
{
	if (pRhs->val.uiVal != 0)
	{
		pResult->val.uiVal = pLhs->val.uiVal % pRhs->val.uiVal;
		pResult->eType = FLM_UINT32_VAL;
	}
	else
	{
		pResult->val.uiVal = 0; // MOD by ZERO case.
		pResult->eType = FLM_UNKNOWN;
	}

	return (FERR_OK);
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC RCODE OpUSMod(
	FQATOM *	pLhs,
	FQATOM *	pRhs,
	FQATOM *	pResult)
{
	if (pRhs->val.iVal != 0)
	{
		pResult->val.iVal = (FLMINT) pLhs->val.uiVal / pRhs->val.iVal;
		pResult->eType = FLM_INT32_VAL;
	}
	else
	{
		pResult->val.uiVal = 0; // MOD by ZERO case
		pResult->eType = FLM_UNKNOWN;
	}

	return (FERR_OK);
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC RCODE OpSSMod(
	FQATOM *	pLhs,
	FQATOM *	pRhs,
	FQATOM *	pResult)
{
	if (pRhs->val.iVal != 0)
	{
		pResult->val.iVal = pLhs->val.iVal % pRhs->val.iVal;
		pResult->eType = (pResult->val.iVal < 0) ? FLM_INT32_VAL : FLM_UINT32_VAL;
	}
	else
	{
		// MOD by ZERO case
		
		pResult->val.uiVal = 0;
		pResult->eType = FLM_UNKNOWN;
	}

	return (FERR_OK);
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC RCODE OpSUMod(
	FQATOM *	pLhs,
	FQATOM *	pRhs,
	FQATOM *	pResult)
{
	if (pRhs->val.uiVal != 0)
	{
		pResult->val.iVal = pLhs->val.iVal % ((FLMINT) pRhs->val.uiVal);
		pResult->eType = FLM_INT32_VAL;
	}
	else
	{
		// Divide by ZERO case.
		
		pResult->val.uiVal = 0;
		pResult->eType = FLM_UNKNOWN;
	}

	return (FERR_OK);
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC RCODE OpUUPlus(
	FQATOM *	pLhs,
	FQATOM *	pRhs,
	FQATOM *	pResult)
{
	pResult->val.uiVal = pLhs->val.uiVal + pRhs->val.uiVal;
	pResult->eType = FLM_UINT32_VAL;
	return (FERR_OK);
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC RCODE OpUSPlus(
	FQATOM *	pLhs,
	FQATOM *	pRhs,
	FQATOM *	pResult)
{
	if ((pRhs->val.iVal >= 0) || (pLhs->val.uiVal > MAX_SIGNED_VAL))
	{
		pResult->val.uiVal = pLhs->val.uiVal + (FLMUINT) pRhs->val.iVal;
		pResult->eType = FLM_UINT32_VAL;
	}
	else
	{
		pResult->val.iVal = (FLMINT) pLhs->val.uiVal + pRhs->val.iVal;
		pResult->eType = (pResult->val.iVal < 0) ? FLM_INT32_VAL : FLM_UINT32_VAL;
	}

	return (FERR_OK);
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC RCODE OpSSPlus(
	FQATOM *	pLhs,
	FQATOM *	pRhs,
	FQATOM *	pResult)
{
	pResult->val.iVal = pLhs->val.iVal + pRhs->val.iVal;
	pResult->eType = (pResult->val.iVal < 0) ? FLM_INT32_VAL : FLM_UINT32_VAL;
	return (FERR_OK);
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC RCODE OpSUPlus(
	FQATOM *	pLhs,
	FQATOM *	pRhs,
	FQATOM *	pResult)
{
	if ((pLhs->val.iVal >= 0) || (pRhs->val.uiVal > MAX_SIGNED_VAL))
	{
		pResult->val.uiVal = (FLMUINT) pLhs->val.iVal + pRhs->val.uiVal;
		pResult->eType = FLM_UINT32_VAL;
	}
	else
	{
		pResult->val.iVal = pLhs->val.iVal + (FLMINT) pRhs->val.uiVal;
		pResult->eType = (pResult->val.iVal < 0) ? FLM_INT32_VAL : FLM_UINT32_VAL;
	}

	return (FERR_OK);
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC RCODE OpURPlus(
	FQATOM *	pLhs,
	FQATOM *	pRhs,
	FQATOM *	pResult)
{
	return (OpSyntaxError( pLhs, pRhs, pResult));
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC RCODE OpRUPlus(
	FQATOM *	pLhs,
	FQATOM *	pRhs,
	FQATOM *	pResult)
{
	return (OpSyntaxError( pLhs, pRhs, pResult));
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC RCODE OpSRPlus(
	FQATOM *	pLhs,
	FQATOM *	pRhs,
	FQATOM *	pResult)
{
	return (OpSyntaxError( pLhs, pRhs, pResult));
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC RCODE OpRSPlus(
	FQATOM *	pLhs,
	FQATOM *	pRhs,
	FQATOM *	pResult)
{
	return (OpSyntaxError( pLhs, pRhs, pResult));
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC RCODE OpRRPlus(
	FQATOM *	pLhs,
	FQATOM *	pRhs,
	FQATOM *	pResult)
{
	return (OpSyntaxError( pLhs, pRhs, pResult));
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC RCODE OpUUMinus(
	FQATOM *	pLhs,
	FQATOM *	pRhs,
	FQATOM *	pResult)
{
	if (pLhs->val.uiVal >= pRhs->val.uiVal)
	{
		pResult->val.uiVal = pLhs->val.uiVal - pRhs->val.uiVal;
		pResult->eType = FLM_UINT32_VAL;
	}
	else
	{
		pResult->val.iVal = (FLMINT) (pLhs->val.uiVal - pRhs->val.uiVal);
		pResult->eType = FLM_INT32_VAL;
	}

	return (FERR_OK);
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC RCODE OpUSMinus(
	FQATOM *	pLhs,
	FQATOM *	pRhs,
	FQATOM *	pResult)
{

	if (pRhs->val.iVal < 0)
	{
		pResult->val.uiVal = pLhs->val.uiVal - pRhs->val.iVal;
		pResult->eType = FLM_UINT32_VAL;
	}
	else
	{
		pResult->val.iVal = (FLMINT) pLhs->val.uiVal - pRhs->val.iVal;
		pResult->eType = (pResult->val.iVal < 0) ? FLM_INT32_VAL : FLM_UINT32_VAL;
	}

	return (FERR_OK);
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC RCODE OpSSMinus(
	FQATOM *	pLhs,
	FQATOM *	pRhs,
	FQATOM *	pResult)
{
	if ((pLhs->val.iVal > 0) && (pRhs->val.iVal < 0))
	{
		pResult->val.uiVal = (FLMUINT) (pLhs->val.iVal - pRhs->val.iVal);
		pResult->eType = FLM_UINT32_VAL;
	}
	else
	{
		pResult->val.iVal = pLhs->val.iVal - pRhs->val.iVal;
		pResult->eType = (pResult->val.iVal < 0) ? FLM_INT32_VAL : FLM_UINT32_VAL;
	}

	return (FERR_OK);
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC RCODE OpSUMinus(
	FQATOM *	pLhs,
	FQATOM *	pRhs,
	FQATOM *	pResult)
{
	if (pRhs->val.uiVal > MAX_SIGNED_VAL)
	{
		pResult->val.iVal = (pLhs->val.iVal - MAX_SIGNED_VAL) - 
										(FLMINT) (pRhs->val.uiVal - MAX_SIGNED_VAL);
		pResult->eType = FLM_INT32_VAL;
	}
	else
	{
		pResult->val.iVal = pLhs->val.iVal - (FLMINT) pRhs->val.uiVal;
		pResult->eType = (pResult->val.iVal < 0) ? FLM_INT32_VAL : FLM_UINT32_VAL;
	}

	return (FERR_OK);
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC RCODE OpURMinus(
	FQATOM *	pLhs,
	FQATOM *	pRhs,
	FQATOM *	pResult)
{
	return (OpSyntaxError( pLhs, pRhs, pResult));
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC RCODE OpRUMinus(
	FQATOM *	pLhs,
	FQATOM *	pRhs,
	FQATOM *	pResult)
{
	return (OpSyntaxError( pLhs, pRhs, pResult));
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC RCODE OpSRMinus(
	FQATOM *	pLhs,
	FQATOM *	pRhs,
	FQATOM *	pResult)
{
	return (OpSyntaxError( pLhs, pRhs, pResult));
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC RCODE OpRSMinus(
	FQATOM *	pLhs,
	FQATOM *	pRhs,
	FQATOM *	pResult)
{
	return (OpSyntaxError( pLhs, pRhs, pResult));
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC RCODE OpRRMinus(
	FQATOM *	pLhs,
	FQATOM *	pRhs,
	FQATOM *	pResult)
{
	return (OpSyntaxError( pLhs, pRhs, pResult));
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE flmCurDoNeg(
	FQATOM *	pResult)
{
	RCODE		rc = FERR_OK;
	FQATOM *	pTmpQAtom;

	// Perform operation on list according to operand types

	for (pTmpQAtom = pResult; pTmpQAtom; pTmpQAtom = pTmpQAtom->pNext)
	{
		if (IS_UNSIGNED( pTmpQAtom->eType))
		{
			if (pTmpQAtom->val.uiVal >= MAX_SIGNED_VAL)
			{
				pTmpQAtom->eType = NO_TYPE;
			}
			else
			{
				pTmpQAtom->val.iVal = -((FLMINT) (pTmpQAtom->val.uiVal));
				pTmpQAtom->eType = FLM_INT32_VAL;
			}
		}
		else if (IS_SIGNED( pTmpQAtom->eType))
		{
			pTmpQAtom->val.iVal *= -1;
		}
		else if (pTmpQAtom->eType != FLM_UNKNOWN)
		{
			rc = RC_SET( FERR_CURSOR_SYNTAX);
			break;
		}
	}

	return (rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMUINT flmCurDoMatchOp(
	FQATOM *	pLhs,
	FQATOM *	pRhs,
	FLMUINT	uiLang,
	FLMBOOL	bLeadingWildCard,
	FLMBOOL	bTrailingWildCard)
{
	FLMUINT	uiFlags = pLhs->uiFlags | pRhs->uiFlags;
	FLMUINT	uiTrueFalse = 0;

	// Verify operand types - non-text and non-binary return false

	if (!IS_BUF_TYPE( pLhs->eType) || !IS_BUF_TYPE( pRhs->eType))
	{
		goto Exit;
	}

	// If one of the operands is binary, simply do a byte comparison of the
	// two values without regard to case or wildcards.

	if ((pLhs->eType == FLM_BINARY_VAL) || (pRhs->eType == FLM_BINARY_VAL))
	{
		FLMUINT	uiLen1;
		FLMUINT	uiLen2;

		uiLen1 = pLhs->uiBufLen;
		uiLen2 = pRhs->uiBufLen;
		flmAssert( !bLeadingWildCard);
		if ((bTrailingWildCard) && (uiLen2 > uiLen1))
		{
			uiLen2 = uiLen1;
		}

		uiTrueFalse = (FLMUINT)
			(
				(
					(uiLen1 == uiLen2) &&
					(f_memcmp( pLhs->val.pucBuf, pRhs->val.pucBuf, uiLen1) == 0)
				) ? (FLMUINT) FLM_TRUE : (FLMUINT) FLM_FALSE
			);
		goto Exit;
	}

	// If wildcards are set, do a string search, first making necessary
	// adjustments for case sensitivity. ;
	//
	// NOTE: THIS IS MATCH BEGIN CASE WITHOUT WILD CARD. The non-wild case
	// for bMatchEntire (DO_MATCH) does NOT come through this section of
	// code. Rather, flmCurDoEQ is called instead of this routine in that
	// case.
	
	if (pLhs->eType == FLM_TEXT_VAL && pRhs->eType == FLM_TEXT_VAL)
	{

		// Always true if there is a wild card.

		uiTrueFalse = flmTextMatch( pLhs->val.pucBuf, pLhs->uiBufLen,
											pRhs->val.pucBuf, pRhs->uiBufLen, uiFlags,
											bLeadingWildCard, bTrailingWildCard, uiLang);
	}
	else
	{
		uiTrueFalse = FLM_FALSE;
	}

Exit:

	return (uiTrueFalse);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMUINT flmCurDoContainsOp(
	FQATOM *	pLhs,
	FQATOM *	pRhs,
	FLMUINT	uiLang)
{
	FLMBYTE *	pResult = NULL;
	FLMUINT		uiFlags = pLhs->uiFlags | pRhs->uiFlags;
	FLMUINT		uiTrueFalse = 0;

	// Verify operands -- both should be buffered types

	if (!IS_BUF_TYPE( pLhs->eType) || !IS_BUF_TYPE( pRhs->eType))
	{
		goto Exit;
	}

	// If one of the operands is binary, simply do a byte comparison of the
	// two values without regard to case or wildcards.

	if ((pLhs->eType == FLM_BINARY_VAL) || (pRhs->eType == FLM_BINARY_VAL))
	{
		uiTrueFalse = FLM_FALSE;
		for (pResult = pLhs->val.pucBuf;
			  (FLMUINT) (pResult - pLhs->val.pucBuf) < pLhs->uiBufLen;
			  pResult++)
		{
			if ((*pResult == pRhs->val.pucBuf[0]) &&
				 (f_memcmp( pLhs->val.pucBuf, pRhs->val.pucBuf, 
						pRhs->uiBufLen) == 0))
			{
				uiTrueFalse = FLM_TRUE;
				goto Exit;
			}
		}

		goto Exit;
	}

	uiTrueFalse = flmTextMatch( pLhs->val.pucBuf, pLhs->uiBufLen,
										pRhs->val.pucBuf, pRhs->uiBufLen, uiFlags,
										TRUE, TRUE, uiLang);
Exit:

	return (uiTrueFalse);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMINT flmCurDoRelationalOp(
	FQATOM *	pLhs,
	FQATOM *	pRhs,
	FLMUINT	uiLang)
{
	FLMUINT	uiFlags = pLhs->uiFlags | pRhs->uiFlags;
	FLMINT	iCompVal = 0;

	switch (pLhs->eType)
	{
		case FLM_TEXT_VAL:
		{
			flmAssert( pRhs->eType == FLM_TEXT_VAL);
			iCompVal = flmTextCompare( pLhs->val.pucBuf, pLhs->uiBufLen,
											  pRhs->val.pucBuf, pRhs->uiBufLen, uiFlags,
											  uiLang);
			break;
		}
		
		case FLM_UINT32_VAL:
		{
			switch (pRhs->eType)
			{
				case FLM_UINT32_VAL:
				{
					iCompVal = FQ_COMPARE( pLhs->val.uiVal, pRhs->val.uiVal);
					break;
				}
				
				case FLM_INT32_VAL:
				{
					if (pRhs->val.iVal < 0)
					{
						iCompVal = 1;
					}
					else
					{
						iCompVal = FQ_COMPARE( pLhs->val.uiVal, 
													  (FLMUINT) pRhs->val.iVal);
					}
					break;
				}
				
				default:
				{
					flmAssert( 0);
					break;
				}

			}
			break;
		}
		
		case FLM_INT32_VAL:
		{
			switch (pRhs->eType)
			{
				case FLM_INT32_VAL:
				{
					iCompVal = FQ_COMPARE( pLhs->val.iVal, pRhs->val.iVal);
					break;
				}
				
				case FLM_UINT32_VAL:
				{
					if (pLhs->val.iVal < 0)
					{
						iCompVal = -1;
					}
					else
					{
						iCompVal = FQ_COMPARE( (FLMUINT) pLhs->val.iVal, 
													  pRhs->val.uiVal);
					}
					break;
				}
				
				default:
				{
					flmAssert( 0);
					break;
				}
			}
			break;
		}
		
		case FLM_REC_PTR_VAL:
		{
			flmAssert( pRhs->eType == FLM_REC_PTR_VAL || 
						  pRhs->eType == FLM_UINT32_VAL);
						  
			iCompVal = FQ_COMPARE( pLhs->val.uiVal, pRhs->val.uiVal);
			break;
		}
		
		case FLM_BINARY_VAL:
		{
			flmAssert( (pRhs->eType == FLM_BINARY_VAL) || 
						  (pRhs->eType == FLM_TEXT_VAL));
						  
			if ((iCompVal = f_memcmp( pLhs->val.pucBuf, pRhs->val.pucBuf, 
						((pLhs->uiBufLen > pRhs->uiBufLen) 
								? pRhs->uiBufLen 
								: pLhs->uiBufLen))) == 0)
			{
				if (pLhs->uiBufLen < pRhs->uiBufLen)
				{
					iCompVal = -1;
				}
				else if (pLhs->uiBufLen > pRhs->uiBufLen)
				{
					iCompVal = 1;
				}
			}
			
			break;
		}
		
		default:
		{
			flmAssert( 0);
			break;
		}
	}

	return (iCompVal);
}
