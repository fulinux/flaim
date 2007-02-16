//-------------------------------------------------------------------------
// Desc: Command line argument parser
// Tabs: 3
//
// Copyright (c) 2001, 2003, 2005-2007 Novell, Inc. All Rights Reserved.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; version 2.1
// of the License.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	See the GNU
// Library Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, contact Novell, Inc.
//
// To contact Novell about this file by physical or electronic mail, 
// you may find current contact information at www.novell.com.
//
// $Id$
//------------------------------------------------------------------------------

#include "ftksys.h"

FSTATIC FLMBOOL matchesAtParams(
	const char *	pszStr);

FSTATIC FLMBOOL matchesEqualsAtParams(
	const char *	pszStr);

/****************************************************************************
Desc:
****************************************************************************/
class F_Arg : public F_Object
{
private:

	F_Arg(
		const char *			pszIdentifier,
		const char *			pszShortHelp,
		FLMBOOL					bCaseSensitive,
		F_ARG_TYPE				argType,
		F_ARG_CONTENT_TYPE	contentType);		
	
	virtual ~F_Arg();

	const char * getIdentifier( void)
	{
		return( m_pszIdentifier);
	}
	
	FLMBOOL isPresent( void)
	{
		return( m_bIsPresent);
	}
	
	FLMUINT getValueCount( void)
	{
		return( m_uiValueCount);
	}
	
	FLMBOOL getCaseSensitive( void)
	{
		return( m_bCaseSensitive);
	}
	
	const char * getShortHelp( void)
	{
		return( m_pszShortHelp);
	}
	
	F_ARG_TYPE getArgType( void)
	{
		return( m_argType);
	}
	
	F_ARG_CONTENT_TYPE getContentType( void)
	{
		return( m_contentType);
	}
	
	F_ARG_VALIDATOR getValidator( void)
	{
		return( m_validator);
	}
	
	void * getValidatorData( void)
	{
		return( m_pvValidatorData);
	}
	
	F_Vector * getStringSet( void)
	{
		return( &m_stringSet);
	}
	
	FLMUINT getStringSetLen( void)
	{
		return( m_uiStringSetCount);
	}
	
	void getMinMax(
		FLMUINT *			puiMin,
		FLMUINT *			puiMax);
		
	void getMinMax( 
		FLMINT *				puiMin,
		FLMINT *				puiMax);

	FLMUINT getUINT(
		FLMUINT				uiIndex);
		
	FLMINT getINT(
		FLMUINT				uiIndex);
		
	FLMBOOL getBOOL(
		FLMUINT				uiIndex);
		
	const char * getString(
		FLMUINT				uiIndex);
		
	void getString(
		char *				pszDestination,
		FLMUINT				uiDestinationBufferSize,
		FLMUINT				uiIndex);

	void setPresent( void)
	{
		m_bIsPresent = TRUE;
	}
	
	RCODE addValue( 
		const char *		pszVal);
		
	const char * getValue(
		FLMUINT				uiIndex);
		
	void setValidator( 
		F_ARG_VALIDATOR	validator,
		void *				pvValidatorData)
	{
		m_validator = validator;
		m_pvValidatorData = pvValidatorData;
	}
	
	void setMinMax(
		FLMUINT				uiMin,
		FLMUINT				uiMax)
	{
		m_uiMin = uiMin;
		m_uiMax = uiMax;
	}
	
	void setMinMax( 
		FLMINT				iMin,
		FLMINT				iMax)
	{
		m_iMin = iMin;
		m_iMax = iMax;
	}
	
	RCODE addToStringSet(
		const char *		pszStr);

	const char *			m_pszIdentifier;
	const char *			m_pszShortHelp;
	FLMBOOL					m_bCaseSensitive;
	F_ARG_TYPE				m_argType;
	F_ARG_CONTENT_TYPE	m_contentType;
	F_Vector					m_valuesVec;
	FLMUINT					m_uiValueCount;
	FLMBOOL					m_bIsPresent;
	F_ARG_VALIDATOR		m_validator;
	void *					m_pvValidatorData;
	FLMUINT					m_uiMin;
	FLMUINT					m_uiMax;
	FLMINT					m_iMin;
	FLMINT					m_iMax;
	F_Vector					m_stringSet;
	FLMUINT					m_uiStringSetCount;
	
friend class F_ArgSet;
};

/****************************************************************************
Desc:
****************************************************************************/
F_Arg::F_Arg(
	const char *			pszIdentifier,
	const char *			pszShortHelp,
	FLMBOOL					bCaseSensitive,
	F_ARG_TYPE				argType,
	F_ARG_CONTENT_TYPE	contentType)		
{
	m_pszIdentifier = pszIdentifier;
	m_pszShortHelp = pszShortHelp;
	m_bCaseSensitive = bCaseSensitive;
	m_argType = argType;
	m_contentType = contentType;
		
	m_uiValueCount = 0;
	m_bIsPresent = FALSE;

	m_validator = NULL;
	m_uiMin = 0xFFFFFFFF;
	m_uiMax = 0xFFFFFFFF;
	m_iMin = -1;
	m_iMax = -1;
	m_uiStringSetCount = 0;
}
	
/****************************************************************************
Desc:
****************************************************************************/
F_Arg::~F_Arg()
{
	FLMUINT		uiKill;
	char *		pszStr;

	for( uiKill = 0; uiKill < m_uiValueCount; uiKill++)
	{
		pszStr = (char *)(m_valuesVec.getElementAt( uiKill));
		
		if( pszStr)
		{
			f_free( &pszStr);
		}
	}
}
	
/****************************************************************************
Desc:
****************************************************************************/
RCODE F_Arg::addValue(
	const char *	pszVal)
{
	RCODE				rc = NE_FLM_OK;
	char *			pszNewVal = NULL;
	
	switch( getContentType())
	{
		case F_ARG_CONTENT_SIGNED_INT:
		{
			// Allow escaped minus, i.e. foo \-1 to pass through a -1
			// and not have it read as an option.	Here, we detect that
			// it made it through, so increment past the '-'
			
			if( pszVal[ 0] == '\\')
			{
				pszVal++;
			}
			
			break;
		}
		
		default:
		{
			break;
		}
	}
	
	if( RC_BAD( rc = f_alloc( f_strlen( pszVal) + 1, &pszNewVal)))
	{
		goto Exit;
	}
	
	f_strcpy( pszNewVal, pszVal);
	
	if( RC_BAD( rc = m_valuesVec.setElementAt( pszNewVal, m_uiValueCount)))
	{
		goto Exit;
	}
	
	pszNewVal = NULL;
	m_uiValueCount++;

Exit:

	if( pszNewVal)
	{
		f_free( &pszNewVal);
	}
	
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
const char * F_Arg::getValue(
	FLMUINT			uiIndex)
{
	f_assert( uiIndex < getValueCount());
	
	return( (const char *)(m_valuesVec.getElementAt( uiIndex)));
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_Arg::addToStringSet(
	const char *		pszStr)
{
	RCODE		rc = NE_FLM_OK;
	
	if( RC_BAD( rc = m_stringSet.setElementAt( 
		(void *)pszStr, m_uiStringSetCount)))
	{
		goto Exit;
	}
	
	m_uiStringSetCount++;
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
void F_Arg::getMinMax( FLMUINT * puiMin, FLMUINT * puiMax)
{
	f_assert( getContentType() == F_ARG_CONTENT_UNSIGNED_INT);
	f_assert( puiMin && puiMax);
	
	*puiMin = m_uiMin;
	*puiMax = m_uiMax;
}

/****************************************************************************
Desc:
****************************************************************************/
void F_Arg::getMinMax( FLMINT * piMin, FLMINT * piMax)
{
	f_assert( getContentType() == F_ARG_CONTENT_SIGNED_INT);
	f_assert( piMin && piMax);
	
	*piMin = m_iMin;
	*piMax = m_iMax;
}

/****************************************************************************
Desc:
****************************************************************************/
FLMUINT F_Arg::getUINT( FLMUINT uiIndex)
{
	return( f_atoud( (const char *)m_valuesVec.getElementAt( uiIndex)));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMINT F_Arg::getINT( FLMUINT uiIndex)
{
	return( f_atoi( (const char *)m_valuesVec.getElementAt( uiIndex)));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMBOOL F_Arg::getBOOL( FLMUINT uiIndex)
{
	return( f_atobool( (const char *)m_valuesVec.getElementAt( uiIndex)));
}

/****************************************************************************
Desc:
****************************************************************************/
const char * F_Arg::getString( FLMUINT uiIndex)
{
	return( (const char *)(m_valuesVec.getElementAt( uiIndex)));
}

/****************************************************************************
Desc:
****************************************************************************/
void F_Arg::getString(
	char *			pszDestination,
	FLMUINT			uiDestinationBufferSize,
	FLMUINT			uiIndex)
{
	const char * 	pszStr = (const char *)m_valuesVec.getElementAt( uiIndex);
	
	f_strncpy( pszDestination, pszStr, uiDestinationBufferSize - 1);
	pszDestination[ uiDestinationBufferSize - 1] = 0;
}

/****************************************************************************
Desc:
****************************************************************************/
F_ArgSet::F_ArgSet(
	char *									pszDescription,
	F_ARG_OUTPUT_CALLBACK				outputCallback,
	void *									pvOutputCallbackData)
{
	m_pszDescription = pszDescription;
	m_outputCallback = outputCallback;
	m_pvOutputCallbackData = pvOutputCallbackData;
	
	m_uiArgVecIndex = 0;
	m_uiArgc = 0;
	m_pArgv = NULL;
	m_pRepeatingArg = NULL;
	m_uiOptionsVecLen = 0;
	m_uiRequiredArgsVecLen = 0;
	m_uiOptionalArgsVecLen = 0;

	m_szExecBaseName[ 0] = 0;
}

/****************************************************************************
Desc:
****************************************************************************/
F_ArgSet::~F_ArgSet()
{
	FLMUINT 		uiKill;
	
	if( m_pArgv)
	{
		// Kill any dynamically allocated memory for the processed
		// command line args
		
		for( uiKill = 0; uiKill < m_uiArgc; uiKill++)
		{
			const char * 	pszStr = (const char *)m_pArgv->getElementAt( uiKill);
			
			if( pszStr)
			{
				f_free( &pszStr);
			}
		}
		
		m_pArgv->Release();
		m_pArgv = NULL;
	}

	// Kill the dynamically allocated F_Arg objs
	
	for( uiKill = 0; uiKill < m_uiArgVecIndex; uiKill++)
	{
		F_Arg *	pArg = (F_Arg *)(m_argVec.getElementAt( uiKill));
		
		if( pArg)
		{
			pArg->Release();
		}
	}
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_ArgSet::addArg(
	const char *			pszIdentifier,
	const char *			pszShortHelp,
	FLMBOOL					bCaseSensitive,
	F_ARG_TYPE				argType,
	F_ARG_CONTENT_TYPE	contentType
	...)
{
	RCODE				rc = NE_FLM_OK;
	FLMUINT			uiLoop;
	const char *	pszExistingStr;
	F_Arg *			pNewArg = NULL;

	// Must have an identifier that's no longer than this, for the
	// printUsage() to display on-screen correctly
	
	f_assert( f_strlen( pszIdentifier) <= 16);

	// Options with argumens need even shorter identifiers
	
	if( (argType == F_ARG_OPTION) && (contentType != F_ARG_CONTENT_NONE))
	{
		f_assert( f_strlen( pszIdentifier) <= 12);
	}

	// Identifier cannot contain '='
	
	f_assert( !f_strchr( pszIdentifier, '='));
	
	// Validate that it's not there already based on the identifier
	
	for( uiLoop = 0; uiLoop < m_uiArgVecIndex; uiLoop++)
	{
		pszExistingStr =
			((F_Arg *)m_argVec.getElementAt( uiLoop))->getIdentifier();
			
		if( f_strcmp( pszExistingStr, pszIdentifier) == 0)
		{
			// Duplicate arg, duplicate identifier found.  If you are
			// designing a utility and you got this error, it means you
			// passed the same first argument to addArg() twice.
			
			f_assert( 0);
		}
	}

	// Enforce order of calling addArg() so it matches the way a utility
	// is called on a command line.  This is so a utility is coded up
	// correctly, since the order matters.
	
	if( argType == F_ARG_REQUIRED_ARG)
	{
		// You have to add required args first, then optionally optional
		// args, then optionally a repeating arg.  The rule is:
		// required args -> optional args -> repeating arg.
		//
		// While all 3 of these types are optional in themselves, you
		// can't have a required arg added after the next two, or
		// an optional arg added after a repeating arg.
		
		f_assert( (m_uiOptionalArgsVecLen == 0) && (!m_pRepeatingArg));
	}
	else if( argType == F_ARG_OPTIONAL_ARG)
	{
		// Can't add an optional arg after adding a repeating arg...the
		// order matters!
		
		f_assert( !m_pRepeatingArg); 
	}

	// Can't have an embedded newline in the short help or it will
	// mess up the word-wrapping in displayShortHelpLines
	
	f_assert( !f_strchr( pszShortHelp, '\n'));

	if( (pNewArg = f_new F_Arg( pszIdentifier, pszShortHelp, bCaseSensitive,
						argType, contentType)) == NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}

	switch( contentType)
	{
		case F_ARG_CONTENT_VALIDATOR:
		{
			f_va_list 			args;
			F_ARG_VALIDATOR 	validator;
			void * 				pvValidatorData;
			
			f_va_start( args, contentType);
			
			validator = f_va_arg( args, F_ARG_VALIDATOR);
			pvValidatorData = f_va_arg( args, void *);
			pNewArg->setValidator( validator, pvValidatorData);
			
			f_va_end( args);
			break;
		}
		
		case F_ARG_CONTENT_SIGNED_INT:
		{
			f_va_list 			args;
			FLMINT 				iMin;
			FLMINT 				iMax;
			
			f_va_start( args, contentType);
			
			iMin = f_va_arg( args, FLMINT);
			iMax = f_va_arg( args, FLMINT);
			pNewArg->setMinMax( iMin, iMax);
			
			f_va_end( args);
			break;
		}
		
		case F_ARG_CONTENT_UNSIGNED_INT:
		{
			f_va_list 			args;
			FLMUINT 				uiMin;
			FLMUINT 				uiMax;
			
			f_va_start( args, contentType);
			
			uiMin = f_va_arg(args, FLMUINT);
			uiMax = f_va_arg(args, FLMUINT);
			
			f_assert( uiMin <= uiMax);
			pNewArg->setMinMax( uiMin, uiMax);
			
			f_va_end( args);
			break;
		}
		
		case F_ARG_CONTENT_ALLOWED_STRING_SET:
		{
			f_va_list 			args;
			
			f_va_start( args, contentType);
			
			for( ;;)
			{
				const char *	pszNext = f_va_arg( args, char *);
				
				if ( !pszNext)
				{
					break;
				}
				
				if( RC_BAD( rc = pNewArg->addToStringSet( pszNext)))
				{
					goto Exit;
				}
			}
			
			f_va_end( args);
		}
		
		case F_ARG_CONTENT_EXISTING_FILE:
		case F_ARG_CONTENT_NONE:
		case F_ARG_CONTENT_BOOL:
		case F_ARG_CONTENT_STRING:
		{
			break;
		}
		
		default:
		{
			f_assert( 0);
			break;
		}
	}

	// Store the specific types of args in their own vector (and pointer
	// in the case of m_pRepeatingArg)
	
	switch( argType)
	{
		case F_ARG_OPTION:
		{
			if( RC_BAD( rc = m_optionsVec.setElementAt(
				pNewArg, m_uiOptionsVecLen)))
			{
				goto Exit;
			}
			
			m_uiOptionsVecLen++;
			break;
		}
		
		case F_ARG_REQUIRED_ARG:
		{
			if( RC_BAD( rc = m_requiredArgsVec.setElementAt(
				pNewArg, m_uiRequiredArgsVecLen)))
			{
				goto Exit;
			}
			
			m_uiRequiredArgsVecLen++;
			break;
		}
		
		case F_ARG_OPTIONAL_ARG:
		{
			if( RC_BAD( rc = m_optionalArgsVec.setElementAt(
				pNewArg, m_uiOptionalArgsVecLen)))
			{
				goto Exit;
			}
			
			m_uiOptionalArgsVecLen++;
			break;
		}
		
		case F_ARG_REPEATING_ARG:
		{
			// Cannot have multiple repeating args
			
			f_assert( !m_pRepeatingArg);
			m_pRepeatingArg = pNewArg;
			break;
		}
		
		default:
		{
			f_assert( 0);
			break;
		}
	}

	// Store all args in a vector
	
	if( RC_BAD( rc = m_argVec.setElementAt( pNewArg, m_uiArgVecIndex)))
	{
		goto Exit;
	}

	pNewArg = NULL;	
	m_uiArgVecIndex++;
	
Exit:

	if( pNewArg)
	{
		pNewArg->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
F_Arg * F_ArgSet::getArg(
	const char *	pszIdentifier)
{
	F_Arg *			pArg;
	FLMUINT			uiLoop;
	
	for( uiLoop = 0; uiLoop < m_uiArgVecIndex; uiLoop++)
	{
		pArg = (F_Arg *)(m_argVec.getElementAt( uiLoop));
		
		if( f_strcmp( pArg->getIdentifier(), pszIdentifier) == 0)
		{
			return( pArg);
		}
	}
	
	f_assert( 0);
	return( NULL);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMBOOL F_ArgSet::needMoreArgs( 
	F_Vector * 		pVec,
	FLMUINT 			uiVecLen)
{
	FLMUINT			uiLoop;
	
	f_assert( pVec);
	
	for( uiLoop = 0; uiLoop < uiVecLen; uiLoop++)
	{
		F_Arg * 		pArg = ((F_Arg *)(pVec->getElementAt( uiLoop)));
		
		// If at least one arg is non-present, we need some more.
		
		if( !pArg->isPresent())
		{
			return( TRUE);
		}
	}
	
	return( FALSE);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_ArgSet::dump( 
	F_Vector * 		pVec, 
	FLMUINT 			uiVecLen)
{
	RCODE 			rc = NE_FLM_OK;
	FLMUINT			uiLoop;
	F_StringAcc 	acc;
	
	// Loop through the args and print out a table for easy reference to see
	// what was set and what wasn't
	
	for( uiLoop = 0; uiLoop < uiVecLen; uiLoop++)
	{
		F_Arg * pArg = (F_Arg *)(pVec->getElementAt( uiLoop));
		
		if( RC_BAD( rc = acc.appendTEXT( "	")))
		{
			goto Exit;
		}
		
		if( RC_BAD( rc = acc.appendTEXT( pArg->getIdentifier())))
		{
			goto Exit;
		}
		
		if( RC_BAD( rc = acc.appendTEXT( ": ")))
		{
			goto Exit;
		}
		
		if( pArg->isPresent())
		{
			if( RC_BAD( rc = acc.appendTEXT( "values={")))
			{
				goto Exit;
			}
					
			for( FLMUINT uiVals = 0; uiVals < pArg->getValueCount(); uiVals++)
			{
				if( RC_BAD( rc = acc.appendTEXT( pArg->getValue( uiVals))))
				{
					goto Exit;
				}
				
				if( uiVals != (pArg->getValueCount() - 1))
				{
					if( RC_BAD( rc = acc.appendTEXT( ",")))
					{
						goto Exit;
					}
				}
			}
			
			if( RC_BAD( rc = acc.appendTEXT( "}\n")))
			{
				goto Exit;
			}
		}
		else
		{
			if( RC_BAD( rc = acc.appendTEXT( "not supplied\n")))
			{
				goto Exit;
			}
		}
	}
	
	if( acc.getTEXT())
	{
		outputLines( (const char *)acc.getTEXT());
	}
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_ArgSet::parseOption( 
	const char *	pszArg, 
	FLMBOOL *		pbPrintedUsage)
{
	RCODE				rc = NE_FLM_OK;
	const char *	pszArgArg;
	FLMBOOL			bFoundMatch = FALSE;
	FLMUINT			uiTraverseOpts;
	
	pszArg++;
	pszArgArg = f_strchr( (const char *)pszArg, '=');
	
	for( uiTraverseOpts = 0; 
		  uiTraverseOpts < m_uiOptionsVecLen;
		  uiTraverseOpts++)
	{
		F_Arg *		pArg = (F_Arg *)m_optionsVec.getElementAt( uiTraverseOpts);
		FLMBOOL		bCaseSensitive = pArg->getCaseSensitive();
		
		if( (bCaseSensitive && (f_strcmp( pArg->getIdentifier(), pszArg) == 0)) ||
			(!bCaseSensitive && (f_stricmp( pArg->getIdentifier(), pszArg) == 0)))
		{
			if( pArg->getContentType() != F_ARG_CONTENT_NONE)
			{
				F_StringAcc acc;
				
				if( RC_BAD( rc = acc.appendTEXT( "ERROR: option ")))
				{
					goto Exit;
				}
					
				if( RC_BAD( rc = acc.appendTEXT( pArg->getIdentifier())))
				{
					goto Exit;
				}
					
				if( RC_BAD( rc = acc.appendTEXT(
					" requires argument of the form option=value")))
				{
					goto Exit;
				}
					
				outputLines( (const char *)acc.getTEXT());
				rc = printUsage();
				*pbPrintedUsage = TRUE;
				
				if( RC_OK( rc))
				{
					rc = RC_SET( NE_FLM_FAILURE);
					goto Exit;
				}
			}
			else
			{
				bFoundMatch = TRUE;
				pArg->setPresent();
				break;
			}
		}
		else if( pszArgArg)
		{
			f_assert( pszArgArg[ 0] == '=');
			
			FLMBOOL bIsMatch =
				((bCaseSensitive &&
					(f_strncmp( pArg->getIdentifier(), pszArg, pszArgArg - pszArg) == 0)) ||
					(!bCaseSensitive &&
						(f_strnicmp( pArg->getIdentifier(), pszArg, pszArgArg - pszArg) == 0)))
				? TRUE
				: FALSE;

			if( bIsMatch)
			{
				pszArgArg++;
				
				if( pArg->getContentType() == F_ARG_CONTENT_NONE)
				{
					F_StringAcc acc;
					
					if( RC_BAD( rc = acc.appendTEXT(
						"ERROR: cannot give argument to option ")))
					{
						goto Exit;
					}
						
					if( RC_BAD( rc = acc.appendTEXT( pArg->getIdentifier())))
					{
						goto Exit;
					}
						
					outputLines( (const char *)acc.getTEXT());
					rc = printUsage();
					*pbPrintedUsage = TRUE;
					
					if( RC_OK( rc))
					{
						rc = RC_SET( NE_FLM_FAILURE);
						goto Exit;
					}
				}
				else
				{
					if( RC_BAD( rc = pArg->addValue( pszArgArg)))
					{
						goto Exit;
					}
					
					pArg->setPresent();
					bFoundMatch = TRUE;
					break;
				}
			}
		}
	}
	
	if ( !bFoundMatch)
	{
		F_StringAcc acc;
		
		if( RC_BAD( rc = acc.appendTEXT( "ERROR: unknown option ")))
		{
			goto Exit;
		}
		
		if( RC_BAD( rc = acc.appendTEXT( pszArg)))
		{
			goto Exit;
		}
		
		outputLines( (const char *)acc.getTEXT());
		rc = printUsage();
		*pbPrintedUsage = TRUE;
		
		if( RC_OK( rc))
		{
			rc = RC_SET( NE_FLM_FAILURE);
		}
		
		goto Exit;
	}
	
Exit:

	return( rc); 
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_ArgSet::parseCommandLine(
	FLMUINT				uiArgc, 
	const char **		ppszArgv,
	FLMBOOL *			pbPrintedUsage)
{
	RCODE					rc = NE_FLM_OK;
	F_StringAcc			errorAcc;
	FLMUINT				uiLoop;
	FLMUINT				uiArgs;
	FLMBOOL				bNegative;
	const char *		pszExecStr = ppszArgv[ 0];
	char					szDir[ F_PATH_MAX_SIZE];
	IF_FileSystem *	pFileSystem = NULL;

	*pbPrintedUsage = FALSE;

	// Set up members we'll need
	
	if( !m_pArgv)
	{
		if( (m_pArgv = f_new F_Vector) == NULL)
		{
			rc = RC_SET( NE_FLM_MEM);
			goto Exit;
		}
	}
	
	// Read file exec basename into variable
	
	if ( RC_BAD( rc = f_pathReduce( pszExecStr, szDir, m_szExecBaseName)))
	{
		goto Exit;
	}
	
	// Copy inital args into member vars
	
	for( uiLoop = 0; uiLoop < uiArgc; uiLoop++)
	{
		const char *	pszCopyThis = ppszArgv[ uiLoop];
		char *			pszNewStr;

		if( RC_BAD( rc = f_alloc( f_strlen( pszCopyThis) + 1, &pszNewStr)))
		{
			goto Exit;
		}

		f_strcpy( pszNewStr, pszCopyThis);
		
		if( RC_BAD( rc = m_pArgv->setElementAt( pszNewStr, uiLoop)))
		{
			f_free( &pszNewStr);
			goto Exit;
		}
		
		m_uiArgc++;
	}
	
	// Pre-process params
	
	if ( RC_BAD( rc = preProcessParams()))
	{
		goto Exit;
	}

	// Now read through the args.  They will all be well-formed at this
	// point.	 Look for strings beginning with a hyphen to find options,
	// and everything else is either a required, optional, or repeating
	// arg.  Ignore arg 0, which is the executable name.
	
	for( uiLoop = 1; uiLoop < m_uiArgc; uiLoop++)
	{
		const char *	pszArg = (const char *)m_pArgv->getElementAt( uiLoop);
		
		f_assert( f_strlen( pszArg) > 0);

		// Handle help strings first
		
		if ( f_stricmp( pszArg, "-h") == 0 ||
			f_stricmp( pszArg, "-help") == 0 ||
			f_strcmp( pszArg, "-?") == 0 ||
			f_strcmp( pszArg, "?") == 0)
		{
			if( RC_BAD( rc = printUsage()))
			{
				goto Exit;
			}
			
			*pbPrintedUsage = TRUE;
			
			if( RC_OK( rc))
			{
				rc = RC_SET( NE_FLM_FAILURE);
			}
			
			goto Exit;
		}
		else if( pszArg[ 0] == '-')
		{
			// Option-handling is sufficiently complex that we'll handle it
			// elsewhere in its own function
			
			if ( RC_BAD( rc = parseOption( pszArg, pbPrintedUsage)))
			{
				goto Exit;
			}
		}
		else if( (m_uiRequiredArgsVecLen > 0) &&
			needMoreArgs( &m_requiredArgsVec, m_uiRequiredArgsVecLen))
		{
			// Required argument
			
			FLMUINT			uiIndex = 0;
			F_Arg *			pChosenArg;
			
			for( ;;)
			{
				pChosenArg = (F_Arg *)m_requiredArgsVec.getElementAt( uiIndex);
				
				if ( !pChosenArg->isPresent())
				{
					break;
				}
				
				uiIndex++;
			}
			
			if( RC_BAD( rc = pChosenArg->addValue( pszArg)))
			{
				goto Exit;
			}
			
			pChosenArg->setPresent();
		}
		else if( (m_uiOptionalArgsVecLen > 0) &&
			needMoreArgs( &m_optionalArgsVec, m_uiOptionalArgsVecLen))
		{
			// Optional argument
			
			FLMUINT			uiIndex = 0;
			F_Arg * 			pChosenArg;
			
			for( ;;)
			{
				pChosenArg = (F_Arg *)m_optionalArgsVec.getElementAt( uiIndex);
				
				if ( !pChosenArg->isPresent())
				{
					break;
				}
				
				uiIndex++;
			}
			
			if( RC_BAD( rc = pChosenArg->addValue( pszArg)))
			{
				goto Exit;
			}
			
			pChosenArg->setPresent();
		}
		else if ( m_pRepeatingArg)
		{
			m_pRepeatingArg->setPresent();
			
			if( RC_BAD( rc = m_pRepeatingArg->addValue( pszArg)))
			{
				goto Exit;
			}
		}
		else
		{
			errorAcc.clear();
			
			if( RC_BAD( rc = errorAcc.appendTEXT( "invalid extra argument ")))
			{
				goto Exit;
			}
			
			if( RC_BAD( rc = errorAcc.appendTEXT( pszArg)))
			{
				goto Exit;
			}
			
			outputLines( (const char *)errorAcc.getTEXT());
			rc = printUsage();
			*pbPrintedUsage = TRUE;
			
			if( RC_OK( rc))
			{
				rc = RC_SET( NE_FLM_FAILURE);
			}
			
			goto Exit;
		}
	}

	// We've read through all the args and set things appropriately.  If
	// we have any required args unset at this point, that's a user
	// problem so let's report it and abort.
	
	if( needMoreArgs( &m_requiredArgsVec, m_uiRequiredArgsVecLen))
	{
		for( uiLoop = 0; uiLoop < m_uiRequiredArgsVecLen; uiLoop++)
		{
			F_Arg * 	pArg = (F_Arg*)m_requiredArgsVec.getElementAt( uiLoop);
			
			if ( !pArg->isPresent())
			{
				errorAcc.clear();
				if( RC_BAD( rc = errorAcc.appendf(
					"ERROR: did not pass required arg #%u <%s>\n",
					uiLoop + 1, pArg->getIdentifier())))
				{
					goto Exit;
				}
			}
		}
		
		outputLines( (const char *)errorAcc.getTEXT());
		rc = printUsage();
		*pbPrintedUsage = TRUE;
		
		if( RC_OK( rc))
		{
			rc = RC_SET( NE_FLM_FAILURE);
		}
		
		goto Exit;
	}

	// Need to loop through all args and validate the inputs.  We can use
	// m_argVec for this (don't need to use the specific groups of args
	// such as m_requiredArgsVec, m_optionalArgsVec, etc.)
	
	for( uiLoop = 0; uiLoop < m_uiArgVecIndex; uiLoop++)
	{
		F_Arg * 		pArg = (F_Arg *)(m_argVec.getElementAt( uiLoop));
		
		f_assert( pArg);
		
		// Set up the accumulator so the generic error message appears
		// prior to the previous one.  Set it up as if will fail ... we may
		// not even use the partial error message.
		
		errorAcc.clear();
		
		// If it's present, then it has a value we will want to validate
		
		if( pArg->isPresent())
		{
			FLMBOOL			bValidated = FALSE;
			const char *	pszStr = "";
			
			switch( pArg->getContentType())
			{
				case F_ARG_CONTENT_NONE:
				case F_ARG_CONTENT_STRING:
				{
					// Nothing to validate for these, they will always match
					
					bValidated = TRUE;
					break;
				}
				
				case F_ARG_CONTENT_BOOL:
				{
					for( uiArgs = 0; uiArgs < pArg->getValueCount(); uiArgs++)
					{
						pszStr = pArg->getValue( uiArgs);
						f_atobool( pszStr, &bValidated);
						
						if( !bValidated)
						{
							if( RC_BAD( errorAcc.appendTEXT( "ERROR: argument '")))
							{
								goto Exit;
							}
							
							if( RC_BAD( errorAcc.appendTEXT( pszStr)))
							{
								goto Exit;
							}
							
							if( RC_BAD( errorAcc.appendTEXT(
								"' is not a valid representation of a boolean")))
							{
								goto Exit;
							}
							
							break;
						}
					}
					
					break;
				}
				
				case F_ARG_CONTENT_VALIDATOR:
				{
					for( uiArgs = 0; uiArgs < pArg->getValueCount(); uiArgs++)
					{
						pszStr = pArg->getValue( uiArgs);
						
						if( !(bValidated = pArg->getValidator()( pszStr, 
									pArg->getIdentifier(), &errorAcc,
									pArg->getValidatorData())))
						{
							break;
						}
					}
					
					break;
				}
				
				case F_ARG_CONTENT_SIGNED_INT:
				case F_ARG_CONTENT_UNSIGNED_INT:
				{
					for( uiArgs = 0; uiArgs < pArg->getValueCount(); uiArgs++)
					{
						pszStr = pArg->getValue( uiArgs);
						
						if( !(bValidated = f_isNumber( pszStr, &bNegative)))
						{
							break;
						}
						
						if( pArg->getContentType() == F_ARG_CONTENT_UNSIGNED_INT &&
							bNegative)
						{
							bValidated = FALSE;
							break;
						}
						
						if( pArg->getContentType() == F_ARG_CONTENT_SIGNED_INT)
						{
							FLMINT 		iMax;
							FLMINT		iMin;
							FLMINT		iArg;
							
							pArg->getMinMax( &iMin, &iMax);
							iArg = f_atoi( pszStr);
							
							if( (iArg > iMax) || (iArg < iMin))
							{
								if( RC_BAD( errorAcc.appendf(
									"ERROR: argument '%s' violates range "
									"requirement of min=%d, max=%d",
									pszStr, iMin, iMax)))
								{
									goto Exit;
								}
										
								bValidated = FALSE;
								break;
							}
						}
						else
						{
							FLMUINT 		uiMax;
							FLMUINT		uiMin;
							FLMUINT		uiArg;
							
							pArg->getMinMax( &uiMin, &uiMax);
							uiArg = f_atoi( pszStr);
							
							if( (uiArg > uiMax) || (uiArg < uiMin))
							{
								if( RC_BAD( errorAcc.appendf(
									"ERROR: argument '%s' violates range "
									"requirement of min=%u, max=%u",
									pszStr, uiMin, uiMax)))
								{
									goto Exit;
								}
								
								bValidated = FALSE;
								break;
							}
						}
					}
					
					break;
				}
				
				case F_ARG_CONTENT_ALLOWED_STRING_SET:
				{
					for( uiArgs = 0; uiArgs < pArg->getValueCount(); uiArgs++)
					{
						FLMUINT 		uiStrSetLen = pArg->getStringSetLen();
						FLMUINT 		uiStrSet;
						FLMBOOL 		bMatched = FALSE;
						F_Vector * 	pStringSet;
						
						pszStr = pArg->getValue( uiArgs);
						pStringSet = pArg->getStringSet();
						
						for( uiStrSet = 0; uiStrSet < uiStrSetLen; uiStrSet++)
						{
							bMatched = (f_strcmp( 
										(const char *)pStringSet->getElementAt( uiStrSet),
										pszStr) == 0) ? TRUE : FALSE;
							if( bMatched)
							{
								break;
							}
						}
						
						bValidated = bMatched;
						
						if( !bValidated)
						{
							if( RC_BAD( rc = errorAcc.appendTEXT( "ERROR: '")))
							{
								goto Exit;
							}
							
							if( RC_BAD( rc = errorAcc.appendTEXT( pszStr)))
							{
								goto Exit;
							}
							
							if( RC_BAD( rc = errorAcc.appendTEXT(
								"' is invalid. Must be a member of {")))
							{
								goto Exit;
							}
							
							for( uiStrSet = 0; uiStrSet < uiStrSetLen; uiStrSet++)
							{
								const char * 	pszNextMember = (const char *)pStringSet->getElementAt( uiStrSet);
								
								if( RC_BAD( rc = errorAcc.appendTEXT( "'")))
								{
									goto Exit;
								}
								
								if( RC_BAD( rc = errorAcc.appendTEXT( pszNextMember)))
								{
									goto Exit;
								}
								
								if( RC_BAD( rc = errorAcc.appendTEXT( "'")))
								{
									goto Exit;
								}
								
								if( uiStrSet != (uiStrSetLen - 1))
								{
									if( RC_BAD( rc = errorAcc.appendTEXT( ",")))
									{
										goto Exit;
									}
								}
							}
							
							if( RC_BAD( rc = errorAcc.appendTEXT( "}\n")))
							{
								goto Exit;
							}
							
							break;
						}
					}
					
					break;
				}
				
				case F_ARG_CONTENT_EXISTING_FILE:
				{
					if( RC_OK( FlmGetFileSystem( &pFileSystem)))
					{
						for( uiArgs = 0; uiArgs < pArg->getValueCount(); uiArgs++)
						{
							pszStr = pArg->getValue( uiArgs);
							bValidated = RC_OK( pFileSystem->doesFileExist( pszStr));
							
							if ( !bValidated)
							{
								if( RC_BAD( rc = errorAcc.appendTEXT( "ERROR: file ")))
								{
									goto Exit;
								}
								
								if( RC_BAD( rc = errorAcc.appendTEXT( pszStr)))
								{
									goto Exit;
								}
								
								if( RC_BAD( rc = errorAcc.appendTEXT( " does not exist\n")))
								{
									goto Exit;
								}
								
								break;
							}
						}
						
						pFileSystem->Release();
						pFileSystem = NULL;
					}
					
					break;
				}
				
				default:
				{
					f_assert( 0);
				}
			}
			
			if( !bValidated)
			{
				if( RC_BAD( rc = errorAcc.appendTEXT( "\nargument '")))
				{
					goto Exit;
				}
				
				if( RC_BAD( rc = errorAcc.appendTEXT( pArg->getIdentifier())))
				{
					goto Exit;
				}
				
				if( RC_BAD( rc = errorAcc.appendTEXT(
					"' did not validate with value '")))
				{
					goto Exit;
				}
				
				if( RC_BAD( rc = errorAcc.appendTEXT( pszStr)))
				{
					goto Exit;
				}
				
				if( RC_BAD( rc = errorAcc.appendTEXT( "'\n")))
				{
					goto Exit;
				}
				
				outputLines( (const char *)errorAcc.getTEXT());
				rc = printUsage();
				*pbPrintedUsage = TRUE;
				
				if( RC_OK( rc))
				{
					rc = RC_SET( NE_FLM_FAILURE);
				}
				
				goto Exit;
			}
		}
	}
	
Exit:

	if( pFileSystem)
	{
		pFileSystem->Release();
	}

	if( RC_BAD( rc) && !(*pbPrintedUsage))
	{
		F_StringAcc acc;
		
		acc.appendf( "ERROR: parseCommandLine(): %u\n", rc);
		outputLines( (const char *)acc.getTEXT());
	}
	
	return( rc);
}

/****************************************************************************
Desc: Print out the short help lines, breaking up at word boundaries.  This
		method has hard-coded agreements with printUsage(), so be careful
		when changing it.
****************************************************************************/
RCODE F_ArgSet::displayShortHelpLines(
	F_StringAcc *	pStringAcc,
	const char *	pszShortHelp,
	FLMUINT			uiCharsPerLine)
{
	RCODE				rc = NE_FLM_OK;
	char *			pszClone = NULL;
	FLMUINT			uiLen = f_strlen( pszShortHelp);
	FLMUINT			uiLoop = 0;
	FLMUINT			uiLastGoodBreakingPos = 0;

	if( RC_BAD( rc = f_strdup( pszShortHelp, &pszClone)))
	{
		goto Exit;
	}
			
	for( ;;)
	{
		if( uiLoop > uiCharsPerLine)
		{
			if ( uiLastGoodBreakingPos == 0)
			{
				// This means that you made words that were too long in the
				// short-help.  You need to break up the words so that
				// this doesn't happen.
				
				f_assert( 0);
				rc = RC_SET( NE_FLM_FAILURE);
				goto Exit;
			}
				
			// Use uiLastGoodBreakingPos to print out a section
			
			pszClone[ uiLastGoodBreakingPos] = 0;
			
			if( RC_BAD( rc = pStringAcc->appendTEXT( pszClone)))
			{
				goto Exit;
			}
			
			if( RC_BAD( rc = pStringAcc->appendTEXT(
				"\n												")))
			{
				goto Exit;
			}
			
			pszClone += f_strlen( pszClone) + 1;
			uiLoop = 0;
			uiLen = f_strlen( pszClone);
			uiLastGoodBreakingPos = 0;
			
			continue;
		}
		else if( uiLoop == (uiLen - 1))
		{
			if( RC_BAD( rc = pStringAcc->appendTEXT( pszClone)))
			{
				goto Exit;
			}
			
			break;
		}
		else if( f_isWhitespace( pszClone[ uiLoop]))
		{
			uiLastGoodBreakingPos = uiLoop;
		}
		
		uiLoop++;
	}
	
Exit:

	if( pszClone)
	{
		f_free( &pszClone);
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_ArgSet::printUsage( void)
{
	RCODE				rc = NE_FLM_OK;
	F_StringAcc		acc;
	FLMUINT			uiLoop = 0;

	// Print usage line
	
	if( RC_BAD( rc = acc.appendTEXT( "Usage: ")))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = acc.appendTEXT( m_szExecBaseName)))
	{
		goto Exit;
	}
	
	if( m_uiOptionsVecLen > 0)
	{
		if( RC_BAD( rc = acc.appendTEXT( " [OPTIONS]")))
		{
			goto Exit;
		}
	}
	
	if( m_uiRequiredArgsVecLen > 0)
	{
		if( RC_BAD( rc = acc.appendTEXT( " ")))
		{
			goto Exit;
		}
	}
	
	for( uiLoop = 0; uiLoop < m_uiRequiredArgsVecLen; uiLoop++)
	{
		F_Arg * 	pArg = (F_Arg*)m_requiredArgsVec.getElementAt( uiLoop);
		
		if( RC_BAD( rc = acc.appendTEXT( "<")))
		{
			goto Exit;
		}
		
		if( RC_BAD( rc = acc.appendTEXT( pArg->getIdentifier())))
		{
			goto Exit;
		}
		
		if( RC_BAD( rc = acc.appendTEXT( ">")))
		{
			goto Exit;
		}
		
		if( uiLoop != (m_uiRequiredArgsVecLen - 1))
		{
			if( RC_BAD( rc = acc.appendTEXT( " ")))
			{
				goto Exit;
			}
		}
	}
	
	if( m_uiOptionalArgsVecLen > 0)
	{
		if( RC_BAD( rc = acc.appendTEXT( " ")))
		{
			goto Exit;
		}
	}
	
	for ( uiLoop = 0; uiLoop < m_uiOptionalArgsVecLen; uiLoop++)
	{
		F_Arg * pArg = (F_Arg*)m_optionalArgsVec.getElementAt( uiLoop);
		
		if( RC_BAD( rc = acc.appendTEXT( "[")))
		{
			goto Exit;
		}
		
		if( RC_BAD( rc = acc.appendTEXT( pArg->getIdentifier())))
		{
			goto Exit;
		}
		
		if( RC_BAD( rc = acc.appendTEXT( "]")))
		{
			goto Exit;
		}
		
		if( uiLoop != (m_uiOptionalArgsVecLen - 1))
		{
			if( RC_BAD( rc = acc.appendTEXT( " ")))
			{
				goto Exit;
			}
		}
	}
	
	if ( m_pRepeatingArg)
	{
		if( RC_BAD( rc = acc.appendTEXT( " [")))
		{
			goto Exit;
		}
		
		if( RC_BAD( rc = acc.appendTEXT( m_pRepeatingArg->getIdentifier())))
		{
			goto Exit;
		}
		
		if( RC_BAD( rc = acc.appendTEXT( "...]")))
		{
			goto Exit;
		}
	}
	
	if( RC_BAD( rc = acc.appendTEXT( "\n")))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = acc.appendTEXT( m_pszDescription)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = acc.appendTEXT( "\n\n")))
	{
		goto Exit;
	}

	if( m_uiOptionsVecLen > 0)
	{
		// The following is fairly hard-coded to get the spacing right
		
		if( RC_BAD( rc = acc.appendTEXT(
			"OPTIONS:\n"
			"\n"
			"identifier			  case sensitive	 description\n"
			"-----------------  --------------	 ---------------------------------------\n")))
		{
			goto Exit;
		}

		// Show all the options
		
		for( uiLoop = 0; uiLoop < m_uiOptionsVecLen; uiLoop++)
		{
			const char *	pszHelpClone = NULL;
			F_Arg *			pArg = (F_Arg*)m_optionsVec.getElementAt( uiLoop);
			const char *	pszIdentifier = pArg->getIdentifier();
			FLMUINT			uiIdentifierLength = f_strlen( pszIdentifier);
			
			if( RC_BAD( rc = acc.appendTEXT( "-")))
			{
				goto Exit;
			}
			
			if( RC_BAD( rc = acc.appendTEXT( pszIdentifier)))
			{
				goto Exit;
			}
			
			if( pArg->getContentType() != F_ARG_CONTENT_NONE)
			{
				const char * 	pszArgArgIndicator = "=ARG";
				
				if( RC_BAD( rc = acc.appendTEXT( pszArgArgIndicator)))
				{
					goto Exit;
				}
				
				uiIdentifierLength += f_strlen( pszArgArgIndicator);
			}
			
			if( RC_BAD( rc = acc.appendCHAR( ' ', (16 - uiIdentifierLength))))
			{
				goto Exit;
			}
			
			if( RC_BAD( rc = acc.appendTEXT( "	")))
			{
				goto Exit;
			}
			
			if( RC_BAD( rc = acc.appendTEXT( (pArg->getCaseSensitive())
																		? "y"
																		: "n")))
			{
				goto Exit;
			}
			
			if( RC_BAD( rc = acc.appendCHAR( ' ', 16)))
			{
				goto Exit;
			}
			
			if( RC_BAD( rc = f_strdup( pArg->getShortHelp(), 
				(char **)&pszHelpClone)))
			{
				goto Exit;
			}
			
			rc = displayShortHelpLines( &acc, pszHelpClone, 39);
			
			if( pszHelpClone)
			{
				f_free( &pszHelpClone);
				pszHelpClone = NULL;
			}
			
			if( RC_BAD( rc))
			{
				goto Exit;
			}
			
			if( RC_BAD( rc = acc.appendTEXT( "\n")))
			{
				goto Exit;
			}
		}
		
		if( RC_BAD( rc = acc.appendTEXT( "\n")))
		{
			goto Exit;
		}
	}

	if( m_uiRequiredArgsVecLen > 0)
	{
		if( RC_BAD( rc = acc.appendTEXT(
			"REQUIRED args:\n"
			"\n")))
		{
			goto Exit;
		}
		
		for( uiLoop = 0; uiLoop < m_uiRequiredArgsVecLen; uiLoop++)
		{
			F_Arg *		pArg = (F_Arg*)m_requiredArgsVec.getElementAt( uiLoop);
			F_StringAcc tempAcc;
			FLMUINT 		uiPads;
			
			if( RC_BAD( rc = tempAcc.appendTEXT( "<")))
			{
				goto Exit;
			}
			
			if( RC_BAD( rc = tempAcc.appendTEXT( pArg->getIdentifier())))
			{
				goto Exit;
			}
			
			if( RC_BAD( rc = tempAcc.appendTEXT( ">: ")))
			{
				goto Exit;
			}
			
			uiPads = 36 - f_strlen( tempAcc.getTEXT());
			
			if( RC_BAD( rc = tempAcc.appendCHAR( ' ', uiPads)))
			{
				goto Exit;
			}
			
			if( RC_BAD( rc = acc.appendTEXT( tempAcc.getTEXT())))
			{
				goto Exit;
			}
			
			if( RC_BAD( rc = displayShortHelpLines(
				&acc, pArg->getShortHelp(), 39)))
			{
				goto Exit;
			}
			
			if( RC_BAD( rc = acc.appendTEXT( "\n")))
			{
				goto Exit;
			}
		}
		
		if( RC_BAD( rc = acc.appendTEXT( "\n\n")))
		{
			goto Exit;
		}
	}

	if ( m_uiOptionalArgsVecLen > 0)
	{
		if( RC_BAD( rc = acc.appendTEXT(
			"OPTIONAL args:\n"
			"\n")))
		{
			goto Exit;
		}
		
		for( uiLoop = 0; uiLoop < m_uiOptionalArgsVecLen; uiLoop++)
		{
			F_Arg *			pArg = (F_Arg*)m_optionalArgsVec.getElementAt( uiLoop);
			F_StringAcc 	tempAcc;
			FLMUINT 			uiPads;
			
			if( RC_BAD( rc = tempAcc.appendTEXT( "[")))
			{
				goto Exit;
			}
			
			if( RC_BAD( rc = tempAcc.appendTEXT( pArg->getIdentifier())))
			{
				goto Exit;
			}
			
			if( RC_BAD( rc = tempAcc.appendTEXT( "]: ")))
			{
				goto Exit;
			}
			
			uiPads = 36 - f_strlen( tempAcc.getTEXT());
			
			if( RC_BAD( rc = tempAcc.appendCHAR( ' ', uiPads)))
			{
				goto Exit;
			}
			
			if( RC_BAD( rc = acc.appendTEXT( tempAcc.getTEXT())))
			{
				goto Exit;
			}
			
			if( RC_BAD( rc = displayShortHelpLines(
				&acc, pArg->getShortHelp(), 39)))
			{
				goto Exit;
			}
			
			if( RC_BAD( rc = acc.appendTEXT( "\n")))
			{
				goto Exit;
			}
		}
		
		if( RC_BAD( rc = acc.appendTEXT( "\n\n")))
		{
			goto Exit;
		}
	}

	if( m_pRepeatingArg)
	{
		F_StringAcc		tempAcc;
		FLMUINT			uiPads;
		
		if( RC_BAD( rc = acc.appendTEXT(
			"REPEATING arg:\n"
			"\n")))
		{
			goto Exit;
		}
		
		if( RC_BAD( rc = tempAcc.appendTEXT( "[")))
		{
			goto Exit;
		}
		
		if( RC_BAD( rc = tempAcc.appendTEXT( m_pRepeatingArg->getIdentifier())))
		{
			goto Exit;
		}
		
		if( RC_BAD( rc = tempAcc.appendTEXT( "...]: ")))
		{
			goto Exit;
		}
		
		uiPads = 36 - f_strlen( tempAcc.getTEXT());
		
		if( RC_BAD( rc = tempAcc.appendCHAR( ' ', uiPads)))
		{
			goto Exit;
		}
		
		if( RC_BAD( rc = acc.appendTEXT( tempAcc.getTEXT())))
		{
			goto Exit;
		}
		
		if( RC_BAD( rc = displayShortHelpLines(
			&acc, m_pRepeatingArg->getShortHelp(), 39)))
		{
			goto Exit;
		}
		
		if( RC_BAD( rc = acc.appendTEXT( "\n\n")))
		{
			goto Exit;
		}
	}
	
	outputLines( (const char *)acc.getTEXT());
	
Exit:

	return( rc);
}

/****************************************************************************
Desc: Output lines of text in the argument by making calls to the
		output callback, one per line.
****************************************************************************/
void F_ArgSet::outputLines( 
	const char *		pszStr)
{
	if( m_outputCallback)
	{
		char				pszOneLine[ 80];
		FLMUINT			uiStrLen = f_strlen( pszStr);
		const char *	pEndLine;

		for( ;;)
		{
			pEndLine = f_strchr( (const char *)pszStr, '\n');
			
			if( uiStrLen == 0)
			{
				break;
			}
			else if( pEndLine && ((pEndLine - pszStr) > 79))
			{
				f_strncpy( pszOneLine, pszStr, 79);
				pszOneLine[ 79] = 0;
				m_outputCallback( pszOneLine, m_pvOutputCallbackData);
				pszStr += 79;
			}
			else if( !pEndLine)
			{
				m_outputCallback( pszStr, m_pvOutputCallbackData);
				break;
			}
			else
			{
				f_strncpy( pszOneLine, pszStr, 79);
				pszOneLine[ 79] = 0;
				m_outputCallback( pszOneLine, m_pvOutputCallbackData);
				
				pszStr = pEndLine + 1;
				uiStrLen = f_strlen( pszStr);
			}
		}
	}
}

/****************************************************************************
Desc: Does the given arg begin with an '@'?
****************************************************************************/
FSTATIC FLMBOOL matchesAtParams( 
	const char *		pszStr)
{
	f_assert( pszStr);
	f_assert( *pszStr);
	
	return( (pszStr[ 0] == '@') ? TRUE : FALSE);
}

/****************************************************************************
Desc: Does the given arg match an "=@filename" type argument
****************************************************************************/
FSTATIC FLMBOOL matchesEqualsAtParams( 
	const char *		pszStr)
{
	const char *		pszStartPos;
	
	f_assert( pszStr);
	f_assert( *pszStr);

	pszStartPos = f_strstr( pszStr, "=@");
	
	if( !pszStartPos)
	{
		return( FALSE);
	}
	else
	{
		// Should have something after the @
		
		if( (f_strlen( pszStartPos) > 2) && pszStartPos[ 2])
		{
			return( TRUE);
		}
		else
		{
			return( FALSE);
		}
	}
}

/****************************************************************************
Desc: Are we done preprocessing the arg set or not for all the @'s?
****************************************************************************/
FLMBOOL F_ArgSet::needsPreprocessing( void)
{
	FLMUINT		uiLoop;
	
	for( uiLoop = 0; uiLoop < m_uiArgc; uiLoop++)
	{
		if( matchesAtParams( (const char *)(m_pArgv->getElementAt( uiLoop))) ||
			 matchesEqualsAtParams( (const char *)(m_pArgv->getElementAt( uiLoop))))
		{
			return( TRUE);
		}
	}
	
	return( FALSE);
}

/****************************************************************************
Desc: Replace all @params.txt with whitespace-collapsed inserted arguments
****************************************************************************/
RCODE F_ArgSet::processAtParams(
	FLMUINT			uiInsertionPoint,
	char *			pszBuffer)
{
	RCODE				rc = NE_FLM_OK;
	F_Vector *		pNewVec = NULL;
	FLMUINT			uiLoop;
	FLMUINT			uiOldVecIndex;
	FLMUINT			uiNewVecIndex = uiInsertionPoint;
	FLMUINT			uiLeftToDo;

	// Set up a new vector and start copying/expanding things over
	
	pNewVec = f_new F_Vector;
	if( !pNewVec)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}

	// Copy over the old items up to the insertion point
	
	for( uiOldVecIndex = 0; uiOldVecIndex < uiInsertionPoint; uiOldVecIndex++)
	{
		const char *	pszOldArg = (const char *)m_pArgv->getElementAt( uiOldVecIndex);
		char *			pszCopy;
		
		if( RC_BAD( rc = f_strdup( pszOldArg, &pszCopy)))
		{
			goto Exit;
		}
		
		if( RC_BAD( rc = pNewVec->setElementAt( pszCopy, uiOldVecIndex)))
		{
			f_free( &pszCopy);
			goto Exit;
		}
	}

	// Handle empty file case
	
	if( pszBuffer)
	{
		char *	pszStartPos;
		char *	pszEndPos;
		
		// Advance pszBuffer to first non-whitespace string
		
		while( f_isWhitespace( *pszBuffer))
		{
			pszBuffer++;
		}
		
		pszStartPos = pszEndPos = pszBuffer;
		
		while( *pszStartPos)
		{
			if( f_isWhitespace( *pszEndPos) || (pszEndPos[ 0] == 0))
			{
				FLMBYTE		ucEndChar = pszEndPos[ 0];
				char * 		pszCopy;
				
				*pszEndPos = 0;
				if( RC_BAD( rc = f_strdup( pszStartPos, &pszCopy)))
				{
					goto Exit;
				}
				
				if( RC_BAD( rc = pNewVec->setElementAt(
					pszCopy, uiNewVecIndex)))
				{
					f_free( &pszCopy);
					goto Exit;
				}
				
				uiNewVecIndex++;

				// If not at end of file, advance through whitespace
				
				if( ucEndChar != 0)
				{
					while( f_isWhitespace( *(++pszEndPos)))
					{
						; // Just advance it to next non-whitespace
					}
				}
				
				pszStartPos = pszEndPos;
			}
			else
			{
				pszEndPos++;
			}
		}
	}

	// Copy the remaining args to the new vector
	
	uiLeftToDo = m_uiArgc - (uiInsertionPoint+1);
	
	for( uiLoop = 0; uiLoop < uiLeftToDo; uiLoop++)
	{
		char *		pszCopy;
		
		if( RC_BAD( rc = f_strdup( (const char *)
			(m_pArgv->getElementAt( uiInsertionPoint + uiLoop + 1)), &pszCopy)))
		{
			goto Exit;
		}
		
		if( RC_BAD( rc = pNewVec->setElementAt( pszCopy, uiNewVecIndex)))
		{
			f_free( &pszCopy);
			goto Exit;
		}
		
		uiNewVecIndex++;
	}

	// Switch over member variables to the new values
	
	for( uiLoop = 0; uiLoop < m_uiArgc; uiLoop++)
	{
		char * 	pszFreeMe = (char *)(m_pArgv->getElementAt( uiLoop));
		
		if( pszFreeMe)
		{
			f_free( &pszFreeMe);
		}
	}
	
	m_pArgv->Release();
	m_pArgv = pNewVec;
	m_uiArgc = uiNewVecIndex;
	
Exit:

	if( RC_BAD( rc))
	{
		FLMUINT		uiClean;
		
		for( uiClean = 0; uiClean < uiNewVecIndex; uiClean++)
		{
			char * 	pszStr = (char *)pNewVec->getElementAt( uiClean);
			
			if( pszStr)
			{
				f_free( &pszStr);
			}
		}
		
		if( pNewVec)
		{
			pNewVec->Release();
		}
	}
	
	return( rc);
}

/****************************************************************************
Desc: Preprocess all @-style arguments
****************************************************************************/
RCODE F_ArgSet::preProcessParams( void)
{
	RCODE					rc = NE_FLM_OK;
	char *				pszBuffer = NULL;
	FLMUINT				uiLoop = 0;
	const FLMUINT		uiMaxPreProcessingCycles = 64;
	FLMUINT				uiPreProcessingCycles = 0;

	for( uiLoop = 0; uiLoop < m_uiArgc;)
	{
		const char * pszNextArg = (const char *)m_pArgv->getElementAt( uiLoop);
		
		if( matchesAtParams( pszNextArg))
		{
			uiPreProcessingCycles++;
			
			if( uiPreProcessingCycles >= uiMaxPreProcessingCycles)
			{
				// There's probably a cycle in the @-files if we're going this deep
				
				outputLines( "ERROR: cycle in @-files detected");
				rc = RC_SET( NE_FLM_FAILURE);
				goto Exit;
			}
			
			// pszNextArg has the form @params.txt
			
			if( RC_BAD( rc = f_filetobuf(
				(const char *)(pszNextArg + 1), &pszBuffer)))
			{
				F_StringAcc		acc;
				
				if( RC_BAD( acc.appendTEXT( "ERROR: reading @-file ")))
				{
					goto Exit;
				}
				
				if( RC_BAD( acc.appendTEXT( pszNextArg)))
				{
					goto Exit;
				}
				
				if( RC_BAD( acc.appendTEXT( "!")))
				{
					goto Exit;
				}
				
				outputLines( (const char *)acc.getTEXT());
				goto Exit;
			}
			
			// f_filetobuf just allocated a pszBuffer with the
			// file contents
			
			rc = processAtParams( uiLoop, pszBuffer);
			
			if( pszBuffer)
			{
				f_free( &pszBuffer);
			}
			
			if( RC_BAD( rc))
			{
				goto Exit;
			}
		}
		else
		{
			uiLoop++;
		}
	}
	
	// Now do the -arg=@params style, once each
	
	for( uiLoop = 0; uiLoop < m_uiArgc; uiLoop++)
	{
		const char * pszNextArg = (const char *)m_pArgv->getElementAt( uiLoop);
		
		if( matchesEqualsAtParams( pszNextArg))
		{
			char *		pszFile = f_strstr( pszNextArg, "=@");
			char *		pszFinalBuffer;

			pszFile++;
			pszFile[ 0] = 0;
			pszFile++;
			
			if( RC_BAD( rc = f_filetobuf( (const char *)pszFile, &pszBuffer)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = f_alloc( f_strlen( pszNextArg) + 
					f_strlen( pszBuffer) + 1, &pszFinalBuffer)))
			{
				f_free( &pszBuffer);
				goto Exit;
			}

			// Replace existing -arg=@params.txt string
			// with one of the form -arg=1 2 3 4 5, etc.
			
			f_strcpy( pszFinalBuffer, pszNextArg);
			f_strcat( pszFinalBuffer, pszBuffer);
			f_free( &pszBuffer);
			
			const char *	pszOldStr = (const char *)m_pArgv->getElementAt( uiLoop);
			
			if( pszOldStr)
			{
				f_free( &pszOldStr);
			}
			
			if( RC_BAD( rc = m_pArgv->setElementAt( pszFinalBuffer, uiLoop)))
			{
				f_free( &pszFinalBuffer);
				goto Exit;
			}
			
			break;
		}
	}
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMBOOL FLMAPI F_ArgSet::argIsPresent( 
	const char *				pszIdentifier)
{
	return( getArg( pszIdentifier)->isPresent());
}

/****************************************************************************
Desc:
****************************************************************************/
FLMUINT FLMAPI F_ArgSet::getValueCount( 
	const char *				pszIdentifier)
{
	return( getArg( pszIdentifier)->getValueCount());
}

/****************************************************************************
Desc:
****************************************************************************/
FLMUINT FLMAPI F_ArgSet::getUINT( 
	const char *				pszIdentifier,
	FLMUINT						uiIndex)
{
	return( getArg( pszIdentifier)->getUINT( uiIndex));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMINT FLMAPI F_ArgSet::getINT( 
	const char *				pszIdentifier,
	FLMUINT						uiIndex)
{
	return( getArg( pszIdentifier)->getINT( uiIndex));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMBOOL FLMAPI F_ArgSet::getBOOL(
	const char *				pszIdentifier,
	FLMUINT						uiIndex)
{
	return( getArg( pszIdentifier)->getBOOL( uiIndex));
}

/****************************************************************************
Desc:
****************************************************************************/
const char * FLMAPI F_ArgSet::getString(
	const char *			pszIdentifier,
	FLMUINT					uiIndex)
{
	return( getArg( pszIdentifier)->getString( uiIndex));
}

/****************************************************************************
Desc:
****************************************************************************/
void FLMAPI F_ArgSet::getString( 
	const char *			pszIdentifier,
	char *					pszDestination,
	FLMUINT					uiDestinationBufferSize,
	FLMUINT					uiIndex)
{
	getArg( pszIdentifier)->getString( pszDestination,
		uiDestinationBufferSize, uiIndex);
}

