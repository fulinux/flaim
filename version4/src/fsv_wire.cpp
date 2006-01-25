//-------------------------------------------------------------------------
// Desc:	Wire class - read and parse client request or server response.
// Tabs:	3
//
//		Copyright (c) 1998-2001,2003,2005-2006 Novell, Inc. All Rights Reserved.
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
// $Id: fsv_wire.cpp 12297 2006-01-19 14:59:48 -0700 (Thu, 19 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#include "flaimsys.h"

/****************************************************************************
Desc:	Resets all member variables to their default / initial values.
*****************************************************************************/
void FSV_WIRE::reset( void)
{
	resetCommon();
	m_uiOpSeqNum = 0;
	m_uiClientVersion = 0;
	m_uiAutoTrans = 0;
	m_uiMaxLockWait = 0;
	m_puzDictPath = NULL;
	m_puzDictBuf = NULL;
	m_puzFileName = NULL;
	m_pucPassword = NULL;
	m_pDrnList = NULL;
	m_uiAreaId = 0;
	m_pIteratorSelect = NULL;
	m_pIteratorFrom = NULL;
	m_pIteratorWhere = NULL;
	m_pIteratorConfig = NULL;
	m_pSession = NULL;
	m_hIterator = HFCURSOR_NULL;
	m_uiType = 0;
	m_bSendGedcom = FALSE;
}

/****************************************************************************
Desc:	Sets the FSV_SESSSION and determines if GEDCOM is supported by the
		client
*****************************************************************************/
void FSV_WIRE::setSession( FSV_SESN * pSession)
{ 
	m_pSession = pSession;

	/*
	See if GEDCOM is supported by the client
	*/
	
	if( m_pSession && (m_pSession->getFlags() & FCS_SESSION_GEDCOM_SUPPORT))
	{
		m_bSendGedcom = TRUE;
	}
}

/****************************************************************************
Desc:	Reads a client request or server response and sets the appropriate
		member variable values.
*****************************************************************************/
RCODE FSV_WIRE::read( void)
{
	FLMUINT	uiTag;
	FLMUINT	uiCount = 0;
	FLMBOOL	bDone = FALSE;
	RCODE		rc = FERR_OK;

	/*
	Read the opcode.
	*/

	if( RC_BAD( rc = readOpcode()))
	{
		goto Exit;
	}

	/*
	Read the request / response values.
	*/

	for( ;;)
	{
		if( RC_BAD( rc = readCommon( &uiTag, &bDone)))
		{
			goto Exit;
		}

		if( bDone)
		{
			goto Exit;
		}

		/*
		uiTag will be non-zero if readCommon did not understand it.
		*/

		uiCount++;
		if( uiTag)
		{
			switch( (uiTag & WIRE_VALUE_TAG_MASK))
			{
				case WIRE_VALUE_OP_SEQ_NUM:
				{
					if( RC_BAD( rc = readNumber( uiTag, &m_uiOpSeqNum, NULL)))
					{
						goto Exit;
					}
					break;
				}

				case WIRE_VALUE_CLIENT_VERSION:
				{
					if( RC_BAD( rc = readNumber( uiTag, &m_uiClientVersion, NULL)))
					{
						goto Exit;
					}
					break;
				}

				case WIRE_VALUE_DICT_FILE_PATH:
				{
					if( RC_BAD( rc = m_pDIStream->readUTF( m_pPool,
						&m_puzDictPath)))
					{
						goto Exit;
					}
					break;
				}

				case WIRE_VALUE_DICT_BUFFER:
				{
					if( RC_BAD( rc = m_pDIStream->readUTF( m_pPool,
						&m_puzDictBuf)))
					{
						goto Exit;
					}
					break;
				}

				case WIRE_VALUE_PASSWORD:
				{
					if( RC_BAD( rc = m_pDIStream->readBinary( m_pPool,
						&m_pucPassword, NULL)))
					{
						goto Exit;
					}
					break;
				}

				case WIRE_VALUE_TYPE:
				{
					if( RC_BAD( rc = readNumber( uiTag, &m_uiType, NULL)))
					{
						goto Exit;
					}
					break;
				}

				case WIRE_VALUE_AREA_ID:
				{
					if( RC_BAD( rc = readNumber( uiTag, &m_uiAreaId, NULL)))
					{
						goto Exit;
					}
					break;
				}

				case WIRE_VALUE_FILE_NAME:
				{
					if( RC_BAD( rc = m_pDIStream->readUTF( m_pPool,
						&m_puzFileName)))
					{
						goto Exit;
					}
					break;
				}

				case WIRE_VALUE_AUTOTRANS:
				{
					if( RC_BAD( rc = readNumber( uiTag,
						&m_uiAutoTrans, NULL)))
					{
						goto Exit;
					}
					break;
				}

				case WIRE_VALUE_ITERATOR_SELECT:
				{
					if( RC_BAD( rc = m_pDIStream->readHTD(
						m_pPool, 0, 0, &m_pIteratorSelect, NULL)))
					{
						goto Exit;
					}
					break;
				}

				case WIRE_VALUE_ITERATOR_FROM:
				{
					if( RC_BAD( rc = m_pDIStream->readHTD(
						m_pPool, 0, 0, &m_pIteratorFrom, NULL)))
					{
						goto Exit;
					}
					break;
				}

				case WIRE_VALUE_ITERATOR_WHERE:
				{
					if( RC_BAD( rc = m_pDIStream->readHTD(
						m_pPool, 0, 0, &m_pIteratorWhere, NULL)))
					{
						goto Exit;
					}
					break;
				}

				case WIRE_VALUE_ITERATOR_CONFIG:
				{
					if( RC_BAD( rc = m_pDIStream->readHTD(
						m_pPool, 0, 0, &m_pIteratorConfig, NULL)))
					{
						goto Exit;
					}
					break;
				}

				case WIRE_VALUE_MAX_LOCK_WAIT:
				{
					if( RC_BAD( rc = readNumber( uiTag,
						&m_uiMaxLockWait, NULL)))
					{
						goto Exit;
					}
					break;
				}

				default:
				{
					if( RC_BAD( rc = skipValue( uiTag)))
					{
						goto Exit;
					}
					break;
				}
			}
		}
	}

Exit:

	return( rc);
}
