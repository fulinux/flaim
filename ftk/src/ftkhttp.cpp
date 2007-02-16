//------------------------------------------------------------------------------
// Desc:	HTTP support
// Tabs:	3
//
// Copyright (c) 2007 Novell, Inc. All Rights Reserved.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; version 2.1
// of the License.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
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

/****************************************************************************
Desc:
****************************************************************************/
class F_HTTPKeyCompare : public IF_ResultSetCompare
{
	RCODE FLMAPI compare(
		const void *			pvData1,
		FLMUINT					uiLength1,
		const void *			pvData2,
		FLMUINT					uiLength2,
		FLMINT *					piCompare)
	{
		FLMINT					iCompare;
		
		if( uiLength1 < uiLength2)
		{
			if( (iCompare = f_strnicmp( (const char *)pvData1, 
				(const char *)pvData2, uiLength1)) == 0)
			{
				iCompare = 1;
			}
		}
		else if( uiLength1 > uiLength2)
		{
			if( (iCompare = f_strnicmp( (const char *)pvData1, 
				(const char *)pvData2, uiLength2)) == 0)
			{
				iCompare = -1;
			}
		}
		else
		{
			iCompare = f_strnicmp( (const char *)pvData1, 
				(const char *)pvData2, uiLength1);
		}
		
		*piCompare = iCompare;
		return( NE_FLM_OK);
	}
};

/****************************************************************************
Desc:
****************************************************************************/
class F_HTTPHeader : public IF_HTTPHeader
{
public:

	F_HTTPHeader()
	{
		m_pResultSet = NULL;
		m_pszRequestURL = NULL;
		resetHeader();
	}
	
	virtual ~F_HTTPHeader()
	{
		resetHeader();
	}
	
	RCODE FLMAPI readResponseHeader(
		IF_IStream *				pIStream);
		
	RCODE FLMAPI writeRequestHeader(
		IF_OStream *				pOStream);
		
	RCODE FLMAPI getHeaderValue(
		const char *				pszTag,
		F_DynaBuf *					pBuffer);
		
	RCODE FLMAPI setHeaderValue(
		const char *				pszTag,
		const char *				pszValue);
		
	RCODE FLMAPI getHeaderValue(
		const char *				pszTag,
		FLMUINT *					puiValue);
		
	RCODE FLMAPI setHeaderValue(
		const char *				pszTag,
		FLMUINT 						uiValue);
		
	RCODE FLMAPI setMethod(
		eHttpMethod					httpMethod);
		
	eHttpMethod getMethod( void);
		
	FLMUINT FLMAPI getStatusCode( void);
	
	RCODE FLMAPI setRequestURL(
		const char *				pszRequestURL);
		
	void FLMAPI resetHeader( void);
	
private:

	RCODE allocResultSet( void);

	IF_BTreeResultSet *			m_pResultSet;
	FLMUINT							m_uiStatusCode;
	FLMUINT							m_uiContentLength;
	eHttpMethod						m_httpMethod;
	char *							m_pszRequestURL;
	char								m_szHttpVersion[ 32];
};

/*****************************************************************************
Desc:
******************************************************************************/
RCODE FLMAPI FlmAllocHTTPHeader( 
	IF_HTTPHeader **			ppHTTPHeader)
{
	if( (*ppHTTPHeader = f_new F_HTTPHeader) == NULL)
	{
		return( RC_SET( NE_FLM_MEM));
	}
	
	return( NE_FLM_OK);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_HTTPHeader::allocResultSet( void)
{
	RCODE						rc = NE_FLM_OK;
	F_HTTPKeyCompare *	pHTTPCompare = NULL;
	
	f_assert( !m_pResultSet);
	
	if( (pHTTPCompare = f_new F_HTTPKeyCompare) == NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}
	
	if( RC_BAD( rc = FlmAllocBTreeResultSet( pHTTPCompare, &m_pResultSet)))
	{
		goto Exit;
	}
	
Exit:

	if( pHTTPCompare)
	{
		pHTTPCompare->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FLMAPI F_HTTPHeader::readResponseHeader(
	IF_IStream *		pIStream)
{
	RCODE					rc = NE_FLM_OK;
	F_DynaBuf			lineBuf;
	F_DynaBuf			tokenBuf;
	FLMUINT				uiLineCount = 0;
	const char *		pszLine;
	const char *		pszTagEnd;
	const char *		pszTagValue;
	
	resetHeader();
	
	if( RC_BAD( rc = allocResultSet()))
	{
		goto Exit;
	}
	
	for( ;;)
	{
		if( RC_BAD( rc = FlmReadLine( pIStream, &lineBuf)))
		{
			goto Exit;
		}
		
		if( *(pszLine = (const char *)lineBuf.getBufferPtr()) == 0)
		{
			break;
		}
		else if( (pszTagEnd = f_strstr( pszLine, ":")) != NULL)
		{
			if( !uiLineCount)
			{
				rc = RC_SET_AND_ASSERT( NE_FLM_BAD_HTTP_HEADER);
				goto Exit;
			}
			
			pszTagValue = pszTagEnd + 1;
			
			while( *pszTagValue && *pszTagValue == ASCII_SPACE)
			{
				pszTagValue++;
			}
			
			if( RC_BAD( rc = m_pResultSet->addEntry( (FLMBYTE *)pszLine, 
				(FLMUINT)(pszTagEnd - pszLine), 
				(FLMBYTE *)pszTagValue, f_strlen( pszTagValue))))
			{
				goto Exit;
			}
		}
		else
		{
			const char * 	pszTmp = pszLine;
			
			// Verify the preamble
			
			if( f_strncmp( pszTmp, "HTTP", 4) != 0)
			{
				rc = RC_SET_AND_ASSERT( NE_FLM_BAD_HTTP_HEADER);
				goto Exit;
			}
			pszTmp += 4;
			
			if( *pszTmp != ASCII_SLASH)
			{
				rc = RC_SET( NE_FLM_BAD_HTTP_HEADER);
				goto Exit;
			}
			pszTmp++;
			
			// Get the protocol version
			
			tokenBuf.truncateData( 0);
			while( *pszTmp && *pszTmp != ASCII_SPACE)
			{
				if( RC_BAD( rc = tokenBuf.appendByte( *pszTmp)))
				{
					goto Exit;
				}
				
				pszTmp++;
			}
			
			tokenBuf.appendByte( 0);
			
			// Skip the space
			
			if( *pszTmp)
			{
				pszTmp++;
			}

			// Get the status code

			tokenBuf.truncateData( 0);
			while( *pszTmp && *pszTmp != ASCII_SPACE)
			{
				if( RC_BAD( rc = tokenBuf.appendByte( *pszTmp)))
				{
					goto Exit;
				}
				
				pszTmp++;
			}
			
			tokenBuf.appendByte( 0);
			m_uiStatusCode = f_atoud( (const char *)tokenBuf.getBufferPtr());
			
			// Skip the space
			
			if( *pszTmp)
			{
				pszTmp++;
			}
			
			// Get the status message

			tokenBuf.truncateData( 0);
			while( *pszTmp)
			{
				if( RC_BAD( rc = tokenBuf.appendByte( *pszTmp)))
				{
					goto Exit;
				}
				
				pszTmp++;
			}
			
			tokenBuf.appendByte( 0);
			uiLineCount++;
		}
		
		if( !uiLineCount)
		{
			rc = RC_SET( NE_FLM_BAD_HTTP_HEADER);
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FLMAPI F_HTTPHeader::getHeaderValue(
	const char *				pszTag,
	F_DynaBuf *					pBuffer)
{
	RCODE				rc = NE_FLM_OK;
	FLMBYTE *		pucValue;
	FLMUINT			uiTagLen = f_strlen( pszTag);
	FLMUINT			uiValueLen;
	
	if( !m_pResultSet)
	{
		rc = RC_SET( NE_FLM_NOT_FOUND);
		goto Exit;
	}
	
	if( RC_BAD( rc = m_pResultSet->findEntry( (FLMBYTE *)pszTag, uiTagLen, 
		NULL, 0, &uiValueLen)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = pBuffer->allocSpace( uiValueLen + 1, (void **)&pucValue)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = m_pResultSet->getCurrent( (FLMBYTE *)pszTag, uiTagLen, 
		pucValue, uiValueLen, NULL)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = pBuffer->appendByte( 0)))
	{
		goto Exit;
	}
		
Exit:

	return( rc);
}
		
/****************************************************************************
Desc:
****************************************************************************/
RCODE FLMAPI F_HTTPHeader::setHeaderValue(
	const char *				pszTag,
	const char *				pszValue)
{
	RCODE				rc = NE_FLM_OK;
	FLMUINT			uiTagLen = f_strlen( pszTag);
	FLMUINT			uiValueLen = f_strlen( pszValue);
	
	if( !m_pResultSet)
	{
		if( RC_BAD( rc = allocResultSet()))
		{
			goto Exit;
		}
	}
	
	if( RC_BAD( rc = m_pResultSet->findEntry( (FLMBYTE *)pszTag, uiTagLen, 
		NULL, 0, NULL)))
	{
		if( rc != NE_FLM_NOT_FOUND)
		{
			goto Exit;
		}
		
		if( RC_BAD( rc = m_pResultSet->addEntry( (FLMBYTE *)pszTag, uiTagLen, 
			(FLMBYTE *)pszValue, uiValueLen)))
		{
			goto Exit;
		}
	}
	else
	{
		if( RC_BAD( rc = m_pResultSet->modifyEntry( (FLMBYTE *)pszTag, uiTagLen, 
			(FLMBYTE *)pszValue, uiValueLen)))
		{
			goto Exit;
		}
	}
		
Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FLMAPI F_HTTPHeader::getHeaderValue(
	const char *				pszTag,
	FLMUINT *					puiValue)
{
	RCODE				rc = NE_FLM_OK;
	F_DynaBuf		valueBuf;
	
	if( RC_BAD( rc = getHeaderValue( pszTag, &valueBuf)))
	{
		goto Exit;
	}
	
	*puiValue = f_atoud( (const char *)valueBuf.getBufferPtr());
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FLMAPI F_HTTPHeader::setHeaderValue(
	const char *				pszTag,
	FLMUINT 						uiValue)
{
	RCODE				rc = NE_FLM_OK;
	char				ucValueBuf[ 32];
	
	f_sprintf( ucValueBuf, "%u", (unsigned)uiValue);
	
	if( RC_BAD( rc = setHeaderValue( pszTag, ucValueBuf)))
	{
		goto Exit;
	}
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMUINT FLMAPI F_HTTPHeader::getStatusCode( void)
{
	return( m_uiStatusCode);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FLMAPI F_HTTPHeader::setMethod(
	eHttpMethod					httpMethod)
{
	m_httpMethod = httpMethod;
	return( NE_FLM_OK);
}
		
/****************************************************************************
Desc:
****************************************************************************/
eHttpMethod F_HTTPHeader::getMethod( void)
{
	return( m_httpMethod);
}
	
/****************************************************************************
Desc:
****************************************************************************/
void FLMAPI F_HTTPHeader::resetHeader( void)
{
	if( m_pResultSet)
	{
		m_pResultSet->Release();
		m_pResultSet = NULL;
	}
	
	if( m_pszRequestURL)
	{
		f_free( &m_pszRequestURL);
	}
	
	m_uiStatusCode = 0;
	m_uiContentLength = 0;
	m_httpMethod = METHOD_GET;
	f_strcpy( m_szHttpVersion, "1.0");
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FLMAPI F_HTTPHeader::setRequestURL(
	const char *	pszRequestURL)
{
	RCODE				rc = NE_FLM_OK;
	FLMUINT			uiStrLen = f_strlen( pszRequestURL);
	
	if( m_pszRequestURL)
	{
		f_free( &m_pszRequestURL);
	}

	if( RC_BAD( rc = f_alloc( uiStrLen + 1, &m_pszRequestURL)))
	{
		goto Exit;
	}
	
	f_memcpy( m_pszRequestURL, pszRequestURL, uiStrLen + 1);
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FLMAPI F_HTTPHeader::writeRequestHeader(
	IF_OStream *				pOStream)
{
	RCODE			rc = NE_FLM_OK;
	FLMBYTE		ucTag[ FLM_MAX_KEY_SIZE];
	FLMBYTE 		ucValue[ 512];
	FLMUINT		uiTagLength;
	FLMUINT		uiValueLength;
	FLMUINT		uiFieldCount;
	
	if( !m_pszRequestURL)
	{
		rc = RC_SET( NE_FLM_BAD_HTTP_HEADER);
		goto Exit;
	}
	
	// Output the request
	
	switch( m_httpMethod)
	{
		case METHOD_GET:
			f_printf( pOStream, "GET ");
			break;
		
		case METHOD_POST:
			f_printf( pOStream, "POST ");
			break;
			
		case METHOD_PUT:
			f_printf( pOStream, "PUT ");
			break;
	}
	
	f_printf( pOStream, "%s ", m_pszRequestURL);
	f_printf( pOStream, "HTTP/%s\r\n", m_szHttpVersion);
	
	// Output the header fields
	
	if( m_pResultSet)
	{
		uiTagLength = 0;
		uiFieldCount = 0;
		
		for( ;;)
		{
			if( !uiFieldCount)
			{
				rc = m_pResultSet->getFirst( ucTag, sizeof( ucTag), 
					&uiTagLength, NULL, 0, &uiValueLength);
			}
			else
			{
				rc = m_pResultSet->getNext( ucTag, sizeof( ucTag), 
					&uiTagLength, NULL, 0, &uiValueLength);
			}
			
			if( RC_BAD( rc))
			{
				if( rc == NE_FLM_EOF_HIT)
				{
					rc = NE_FLM_OK;
					break;
				}
				
				goto Exit;
			}
			
			if( uiValueLength >= sizeof( ucValue))
			{
				rc = RC_SET( NE_FLM_BAD_HTTP_HEADER);
				goto Exit;
			}
			
			if( RC_BAD( rc = m_pResultSet->getCurrent( ucTag, uiTagLength, 
				ucValue, uiValueLength, NULL)))
			{
				goto Exit;
			}
			
			ucTag[ uiTagLength] = 0;
			ucValue[ uiValueLength] = 0;
			
			f_printf( pOStream, "%s: %s\r\n", ucTag, ucValue);
			uiFieldCount++;
		}
	}
	
	// Terminate

	f_printf( pOStream, "\r\n");
	
Exit:

	return( rc);
}

