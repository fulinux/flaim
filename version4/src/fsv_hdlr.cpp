//-------------------------------------------------------------------------
// Desc:	Routines needed to service client requests made to the server.
// Tabs:	3
//
//		Copyright (c) 1998-2006 Novell, Inc. All Rights Reserved.
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
// $Id: fsv_hdlr.cpp 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#include "flaimsys.h"

FSTATIC RCODE fsvIteratorParse(
	FSV_WIRE *	pWire,
	POOL *		pPool);

FSTATIC RCODE fsvIteratorWhereParse(
	FSV_WIRE *	pWire,
	POOL *		pPool);

FSTATIC RCODE fsvIteratorFromParse(
	FSV_WIRE *	pWire,
	POOL *		pPool);

FSTATIC RCODE fsvIteratorSelectParse(
	FSV_WIRE *	pWire,
	POOL *		pPool);

FSTATIC RCODE fsvDbGetBlocks(
	HFDB				hDb,
	FLMUINT			uiAddress,
	FLMUINT			uiMinTransId,
	FLMUINT *		puiCount,
	FLMUINT *		puiBlocksExamined,
	FLMUINT *		puiNextBlkAddr,
	FLMUINT			uiFlags,
	POOL *			pPool,
	FLMBYTE **		ppBlocks,
	FLMUINT *		puiBytes);

FSTATIC RCODE fsvGetHandles(
	FSV_WIRE *     pWire);

/*
Function / Method Implementations
*/

/****************************************************************************
Desc:    This is the function that processes FLAIM requests.
*****************************************************************************/
RCODE fsvProcessRequest(
	FCS_DIS *         pDataIStream,
	FCS_DOS *         pDataOStream,
	POOL *				pScratchPool,
	FLMUINT *			puiSessionIdRV)
{
	void *		pvMark = NULL;
	FSV_WIRE    Wire( pDataIStream, pDataOStream);
	RCODE       rc = FERR_OK;

	/*
	Set the temporary pool
	*/

	if( pScratchPool)
	{
		pvMark = GedPoolMark( pScratchPool);
		Wire.setPool( pScratchPool);
	}

	/*
	Read the request
	*/

	if( RC_BAD( rc = Wire.read()))
	{
		goto Exit;
	}

	/*
	Close the input stream.
	*/

	pDataIStream->close();
	Wire.setDIStream( NULL);

	/*
	Get any required handles.
	*/

	if( RC_BAD( rc = fsvGetHandles( &Wire)))
	{
#ifdef FSV_LOGGING
		fsvLogHandlerMessage( NULL,
			(FLMBYTE *)"Error calling fsvGetHandles", rc, FSV_LOG_DEBUG);
#endif
		goto Exit;
	}

	/*
	Call the appropriate handler function.
	*/

	switch( Wire.getClass())
	{
		case FCS_OPCLASS_GLOBAL:
		{
#ifdef FSV_LOGGING
			fsvLogHandlerMessage( NULL,
				(FLMBYTE *)"OpClass: Global", 0, FSV_LOG_DEBUG);
#endif
			if( RC_BAD( rc = fsvOpClassGlobal( &Wire)))
			{
				goto Exit;
			}
			break;
		}

		case FCS_OPCLASS_SESSION:
		{
#ifdef FSV_LOGGING
			fsvLogHandlerMessage( NULL,
				(FLMBYTE *)"OpClass: Session", 0, FSV_LOG_DEBUG);
#endif
			if( RC_BAD( rc = fsvOpClassSession( &Wire)))
			{
				goto Exit;
			}
			break;
		}

		case FCS_OPCLASS_DATABASE:
		{
#ifdef FSV_LOGGING
			fsvLogHandlerMessage( NULL,
				(FLMBYTE *)"OpClass: Database", 0, FSV_LOG_DEBUG);
#endif
			if( RC_BAD( rc = fsvOpClassDatabase( &Wire)))
			{
				goto Exit;
			}
			break;
		}

		case FCS_OPCLASS_TRANS:
		{
#ifdef FSV_LOGGING
			fsvLogHandlerMessage( NULL,
				(FLMBYTE *)"OpClass: Transaction", 0, FSV_LOG_DEBUG);
#endif
			if( RC_BAD( rc = fsvOpClassTransaction( &Wire)))
			{
				goto Exit;
			}
			break;
		}

		case FCS_OPCLASS_RECORD:
		{
#ifdef FSV_LOGGING
			fsvLogHandlerMessage( NULL,
				(FLMBYTE *)"OpClass: Record", 0, FSV_LOG_DEBUG);
#endif
			if( RC_BAD( rc = fsvOpClassRecord( &Wire)))
			{
				goto Exit;
			}
			break;
		}

		case FCS_OPCLASS_ITERATOR:
		{
#ifdef FSV_LOGGING
			fsvLogHandlerMessage( NULL,
				(FLMBYTE *)"OpClass: Iterator", 0, FSV_LOG_DEBUG);
#endif
			if( RC_BAD( rc = fsvOpClassIterator( &Wire)))
			{
				goto Exit;
			}
			break;
		}

		case FCS_OPCLASS_BLOB:
		{
#ifdef FSV_LOGGING
			fsvLogHandlerMessage( NULL,
				(FLMBYTE *)"OpClass: BLOB", 0, FSV_LOG_DEBUG);
#endif
			rc = RC_SET( FERR_NOT_IMPLEMENTED);
			break;
		}

		case FCS_OPCLASS_DIAG:
		{
#ifdef FSV_LOGGING
			fsvLogHandlerMessage( NULL,
				(FLMBYTE *)"OpClass: Diagnostic", 0, FSV_LOG_DEBUG);
#endif
			if( RC_BAD( rc = fsvOpClassDiag( &Wire)))
			{
				goto Exit;
			}
			break;
		}

		case FCS_OPCLASS_FILE:
		{
#ifdef FSV_LOGGING
			fsvLogHandlerMessage( NULL,
				(FLMBYTE *)"OpClass: File System", 0, FSV_LOG_DEBUG);
#endif
			if( RC_BAD( rc = fsvOpClassFile( &Wire)))
			{
				goto Exit;
			}
			break;
		}

		case FCS_OPCLASS_ADMIN:
		{
#ifdef FSV_LOGGING
			fsvLogHandlerMessage( NULL,
				(FLMBYTE *)"OpClass: Admin", 0, FSV_LOG_DEBUG);
#endif
			if( RC_BAD( rc = fsvOpClassAdmin( &Wire)))
			{
				goto Exit;
			}
			break;
		}

		case FCS_OPCLASS_INDEX:
		{
#ifdef FSV_LOGGING
			fsvLogHandlerMessage( NULL,
				(FLMBYTE *)"OpClass: Index", 0, FSV_LOG_DEBUG);
#endif
			if( RC_BAD( rc = fsvOpClassIndex( &Wire)))
			{
				goto Exit;
			}
			break;
		}

		case FCS_OPCLASS_MISC:
		{
#ifdef FSV_LOGGING
			fsvLogHandlerMessage( NULL,
				(FLMBYTE *)"OpClass: Misc.", 0, FSV_LOG_DEBUG);
#endif
			if( RC_BAD( rc = fsvOpClassMisc( &Wire)))
			{
				goto Exit;
			}
			break;
		}

		default:
		{
			rc = RC_SET( FERR_NOT_IMPLEMENTED);
			goto Exit;
		}
	}

	if( puiSessionIdRV)
	{
		// Set the session ID so that the calling routine has
		// the option of performing cleanup on an error
		*puiSessionIdRV = Wire.getSessionId();
	}

Exit:

	if( RC_BAD( rc))
	{
		/*
		If the input stream is still open, the handler never
		send any data to the client.  Close the input stream and
		try to send the error code to the client.
		*/

		if( pDataIStream->isOpen())
		{
			(void)pDataIStream->close();
			Wire.setDIStream( NULL);
		}

		if( RC_OK( Wire.sendOpcode( Wire.getClass(), Wire.getOp())))
		{
			if( RC_OK( Wire.sendRc( rc)))
			{
				if( RC_OK( Wire.sendTerminate()))
				{
					pDataOStream->close();
				}
			}
		}
	}
	else
	{
		pDataOStream->close();
	}

	if( pScratchPool)
	{
		GedPoolReset( pScratchPool, pvMark);
	}

	return( rc);
}


/****************************************************************************
Desc:    Performs a diagnostic operation
*****************************************************************************/
RCODE fsvOpClassDiag(
	FSV_WIRE *		pWire)
{
	RCODE       opRc = FERR_OK;
	RCODE       rc = FERR_OK;

	/*
	Service the request.
	*/

	switch( pWire->getOp())
	{
		case FCS_OP_DIAG_HTD_ECHO:
		{
			/*
			Simply echo the record back to the client.  This
			is done below when the response is sent to the client.
			*/
			break;
		}

		default:
		{
			opRc = RC_SET( FERR_NOT_IMPLEMENTED);
			goto OP_EXIT;
		}
	}


OP_EXIT:

	/*
	Send the server's response.
	*/

	if( RC_BAD( rc = pWire->sendOpcode( FCS_OPCLASS_DIAG, pWire->getOp())))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pWire->sendRc( opRc)))
	{
		goto Exit;
	}

	if( RC_OK( opRc))
	{
		switch( pWire->getOp())
		{
			case FCS_OP_DIAG_HTD_ECHO:
			{
				if( pWire->getRecord() != NULL)
				{
					if( RC_BAD( rc = pWire->sendRecord(
						WIRE_VALUE_HTD, pWire->getRecord())))
					{
						goto Exit;
					}
				}
				break;
			}
		}
	}

	if( RC_BAD( rc = pWire->sendTerminate()))
	{
		goto Exit;
	}

Exit:

	return( rc);
}


/****************************************************************************
Desc:    Performs a file system operation
*****************************************************************************/
RCODE fsvOpClassFile(
	FSV_WIRE *		pWire)
{
	RCODE				rc = FERR_OK;
	RCODE				opRc = FERR_OK;
	FSV_SCTX *		pServerContext = NULL;
	FLMUNICODE *	puzSourcePath;
	char 				szSourcePath[ F_PATH_MAX_SIZE];

 	/*
	Set up local variables.
	*/

	if( RC_BAD( opRc = fsvGetGlobalContext( &pServerContext)))
	{
		goto OP_EXIT;
	}

	puzSourcePath = pWire->getFilePath();
	if( puzSourcePath)
	{
		/*
		Convert the UNICODE URL to a server path.
		*/

		if( RC_BAD( rc = pServerContext->BuildFilePath(
			puzSourcePath, szSourcePath)))
		{
			goto Exit;
		}
	}
	
	/*
	Service the request.
	*/

	switch( pWire->getOp())
	{
		case FCS_OP_FILE_EXISTS:
		{
			if( !puzSourcePath)
			{
				opRc = RC_SET( FERR_SYNTAX);
				goto OP_EXIT;
			}

			if( RC_BAD( opRc = gv_FlmSysData.pFileSystem->Exists( szSourcePath)))
			{
				goto OP_EXIT;
			}

			break;
		}

		case FCS_OP_FILE_DELETE:
		{
			if( !puzSourcePath)
			{
				opRc = RC_SET( FERR_SYNTAX);
				goto OP_EXIT;
			}

			if( RC_BAD( opRc =
				gv_FlmSysData.pFileSystem->Delete( szSourcePath)))
			{
				goto OP_EXIT;
			}

			break;
		}

		default:
		{
			opRc = RC_SET( FERR_NOT_IMPLEMENTED);
			goto OP_EXIT;
		}
	}


OP_EXIT:

	/*
	Send the server's response.
	*/

	if( RC_BAD( rc = pWire->sendOpcode( FCS_OPCLASS_FILE, pWire->getOp())))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pWire->sendRc( opRc)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pWire->sendTerminate()))
	{
		goto Exit;
	}

Exit:

	return( rc);
}


/****************************************************************************
Desc:    Performs an administrative operation
*****************************************************************************/
RCODE fsvOpClassAdmin(
	FSV_WIRE *		pWire)
{
	RCODE       opRc = FERR_OK;
	RCODE       rc = FERR_OK;

	/*
	Service the request.
	*/

//	switch( pWire->getOp())
//	{
//		default:
//		{
			opRc = RC_SET( FERR_NOT_IMPLEMENTED);
			goto OP_EXIT;
//		}
//	}

OP_EXIT:

	/*
	Send the server's response.
	*/

	if( RC_BAD( rc = pWire->sendOpcode( FCS_OPCLASS_ADMIN, pWire->getOp())))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pWire->sendRc( opRc)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pWire->sendTerminate()))
	{
		goto Exit;
	}

Exit:

	return( rc);
}


/****************************************************************************
Desc:    Performs a global operation
*****************************************************************************/
RCODE fsvOpClassGlobal(
	FSV_WIRE *		pWire)
{
	FSV_SCTX *	pServerContext;
	NODE *		pTree = NULL;
	RCODE			opRc = FERR_OK;
	RCODE			rc = FERR_OK;

	/*
	Service the request.
	*/

	if( RC_BAD( rc = fsvGetGlobalContext( &pServerContext)))
	{
		goto Exit;
	}

	switch( pWire->getOp())
	{
		case FCS_OP_GLOBAL_STATS_START:
		{
			if( RC_BAD( opRc = FlmConfig( FLM_START_STATS, 0, 0)))
			{
				goto OP_EXIT;
			}
			break;
		}

		case FCS_OP_GLOBAL_STATS_STOP:
		{
			if( RC_BAD( opRc = FlmConfig( FLM_STOP_STATS, 0, 0)))
			{
				goto OP_EXIT;
			}
			break;
		}

		case FCS_OP_GLOBAL_STATS_RESET:
		{
			if( RC_BAD( opRc = FlmConfig( FLM_RESET_STATS, 0, 0)))
			{
				goto OP_EXIT;
			}
			break;
		}

		case FCS_OP_GLOBAL_MEM_INFO_GET:
		{
			FLM_MEM_INFO	memInfo;

			FlmGetMemoryInfo( &memInfo);
			if( RC_BAD( opRc = fcsBuildMemInfo( &memInfo, 
				pWire->getPool(), &pTree)))
			{
				goto OP_EXIT;
			}
			break;
		}

		case FCS_OP_GLOBAL_GET_THREAD_INFO:
		{
			if( RC_BAD( opRc = fcsBuildThreadInfo( 
				pWire->getPool(), &pTree)))
			{
				goto OP_EXIT;
			}
			break;
		}

		default:
		{
			opRc = RC_SET( FERR_NOT_IMPLEMENTED);
			goto OP_EXIT;
		}
	}


OP_EXIT:

	/*
	Send the server's response.
	*/

	if( RC_BAD( rc = pWire->sendOpcode( FCS_OPCLASS_GLOBAL, pWire->getOp())))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pWire->sendRc( opRc)))
	{
		goto Exit;
	}

	if( RC_OK( opRc))
	{
		if( pTree)
		{
			if( RC_BAD( rc = pWire->sendHTD( WIRE_VALUE_HTD, pTree)))
			{
				goto Exit;
			}
		}
	}

	if( RC_BAD( rc = pWire->sendTerminate()))
	{
		goto Exit;
	}

Exit:

	return( rc);
}


/****************************************************************************
Desc:    Performs a session operation
*****************************************************************************/
RCODE fsvOpClassSession(
	FSV_WIRE *           pWire)
{
	FLMUINT		uiSessionIdRV;
	FSV_SCTX *  pServerContext;
	FSV_SESN *  pSession = NULL;
#ifdef FSV_LOGGING
	char			szLogBuf[ FSV_LOG_BUFFER_SIZE];
#endif
	RCODE			opRc = FERR_OK;
	RCODE			rc = FERR_OK;

#ifdef FSV_LOGGING
	szLogBuf[ 0] = '\0';
#endif

	/*
	Service the request.
	*/

	if( RC_BAD( opRc = fsvGetGlobalContext( &pServerContext)))
	{
		goto OP_EXIT;
	}

	switch( pWire->getOp())
	{
		case FCS_OP_SESSION_OPEN:
		{
#ifdef FSV_LOGGING
			f_sprintf( szLogBuf,
				"SessionOpen: CV = %8.8X, FL = %8.8X",
				(unsigned)pWire->getClientVersion(),
				(unsigned)pWire->getFlags());
#endif

			/*
			Create a new session.
			*/
			
			if( RC_BAD( opRc = pServerContext->OpenSession(
				pWire->getClientVersion(), pWire->getFlags(),
				&uiSessionIdRV, &pSession)))
			{
				goto OP_EXIT;
			}

#ifdef FSV_LOGGING
			f_sprintf( szLogBuf,
				"SessionOpen: ID = %8.8X, CV = %8.8X, FL = %8.8X",
				(unsigned)uiSessionIdRV,
				(unsigned)pWire->getClientVersion(),
				(unsigned)pWire->getFlags());
#endif

			break;
		}

		case FCS_OP_SESSION_CLOSE:
		{
#ifdef FSV_LOGGING
			f_sprintf( szLogBuf,
				"SessionClose: ID = %8.8X",
				(unsigned)pWire->getSessionId());
#endif

			/*
			Close the session.
			*/

			if( RC_BAD( opRc =
				pServerContext->CloseSession( pWire->getSessionId())))
			{
				goto OP_EXIT;
			}
			break;
		}

		default:
		{
			opRc = RC_SET( FERR_NOT_IMPLEMENTED);
			goto OP_EXIT;
		}
	}

OP_EXIT:

	/*
	Send the response.
	*/

	if( RC_BAD( rc = pWire->sendOpcode( FCS_OPCLASS_SESSION, pWire->getOp())))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pWire->sendRc( opRc)))
	{
		goto Exit;
	}

	if( RC_OK( opRc))
	{
		if( pWire->getOp() == FCS_OP_SESSION_OPEN)
		{
			if( RC_BAD( rc = pWire->sendNumber( 
				WIRE_VALUE_SESSION_ID, uiSessionIdRV)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = pWire->sendNumber( WIRE_VALUE_SESSION_COOKIE,
				pSession->getCookie())))
			{
				goto Exit;
			}

			if( RC_BAD( rc = pWire->sendNumber( WIRE_VALUE_FLAGS, 
				FCS_SESSION_GEDCOM_SUPPORT)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = pWire->sendNumber( WIRE_VALUE_FLAIM_VERSION, 
				FLM_CURRENT_VERSION_NUM)))
			{
				goto Exit;
			}
		}
	}

	if( RC_BAD( rc = pWire->sendTerminate()))
	{
		goto Exit;
	}

Exit:

#ifdef FSV_LOGGING
	fsvLogHandlerMessage( NULL, szLogBuf, opRc ? opRc : rc, FSV_LOG_EVENT);
#endif

	return( rc);
}


/****************************************************************************
Desc:    Performs a record or DRN operation
*****************************************************************************/
RCODE fsvOpClassRecord(
	FSV_WIRE *		pWire)
{
	FSV_SESN *	pSession;
	HFDB			hDb;
	FLMUINT		uiContainer;
	FLMUINT		uiIndex;
	FLMUINT		uiAutoTrans;
	FLMUINT		uiDrn;
	FLMUINT		uiFlags;
	FlmRecord *	pRecord = NULL;
	FlmRecord *	pRecordRV = NULL;
	FLMUINT		uiDrnRV = 0;
#ifdef FSV_LOGGING
	char			szLogBuf[ FSV_LOG_BUFFER_SIZE];
#endif
	RCODE       opRc = FERR_OK;
	RCODE       rc = FERR_OK;

#ifdef FSV_LOGGING
	szLogBuf[ 0] = 0;
#endif

	/*
	Get a pointer to the session object.
	*/

	if( (pSession = pWire->getSession()) == NULL)
	{
		opRc = RC_SET( FERR_BAD_HDL);
		goto OP_EXIT;
	}

	/*
	Get the database handle.  This is needed by all of the
	record operations.
	*/
	
	if( (hDb = (HFDB)pWire->getFDB()) == HFDB_NULL)
	{
		opRc = RC_SET( FERR_BAD_HDL);
		goto OP_EXIT;
	}

	/*
	Initialize local variables.
	*/

	uiContainer = pWire->getContainerId();
	uiIndex = pWire->getIndexId();
	uiDrn = pWire->getDrn();
	uiAutoTrans = pWire->getAutoTrans();
	uiFlags = pWire->getFlags();
	pRecord = pWire->getRecord();

	/*
	Perform the operation.
	*/
	
	switch( pWire->getOp())
	{
		case FCS_OP_RECORD_RETRIEVE:
		{
#ifdef FSV_LOGGING
			f_sprintf( szLogBuf,
				"RecRtrv: CO = %4.4X, DRN = %8.8X",
				(unsigned)uiContainer, (unsigned)uiDrn);
#endif
			if( !uiFlags)
			{
				uiFlags = FO_EXACT;
			}

			if( pWire->getBoolean())
			{
				/*
				Fetch the record
				*/

				if( RC_BAD( opRc = FlmRecordRetrieve( hDb,
					uiContainer, uiDrn, uiFlags, &pRecordRV, &uiDrnRV)))
				{
					goto OP_EXIT;
				}
			}
			else
			{
				/*
				Just get the DRN
				*/

				if( RC_BAD( opRc = FlmRecordRetrieve( hDb,
					uiContainer, uiDrn, uiFlags, NULL, &uiDrnRV)))
				{
					goto OP_EXIT;
				}
			}
			break;
		}

		case FCS_OP_RECORD_ADD:
		{
#ifdef FSV_LOGGING
			f_sprintf( szLogBuf,
				"RecAdd: CO = %4.4X, DRN = %8.8X",
				(unsigned)uiContainer, (unsigned)uiDrn);
#endif

			uiDrnRV = uiDrn;
			if( RC_BAD( opRc = FlmRecordAdd( hDb,
				uiContainer,
				&uiDrnRV,
				pRecord,
				uiAutoTrans)))
			{
				goto OP_EXIT;
			}

#ifdef FSV_LOGGING
			/*
			Need to change the log buffer after the operation so that
			it correctly represents the outcome of the operation.
			*/

			f_sprintf( szLogBuf,
				"RecAdd: CO = %4.4X, DRN = %8.8X",
				(unsigned)uiContainer, (unsigned)uiDrnRV);
#endif

			break;
		}

		case FCS_OP_RECORD_MODIFY:
		{
#ifdef FSV_LOGGING
			f_sprintf( szLogBuf,
				"RecMod: CO = %4.4X, DRN = %8.8X",
				(unsigned)uiContainer, (unsigned)uiDrn);
#endif

			if( RC_BAD( opRc = FlmRecordModify( hDb,
				uiContainer,
				uiDrn,
				pRecord,
				uiAutoTrans)))
			{
				goto OP_EXIT;
			}
			break;
		}
		
		case FCS_OP_RECORD_DELETE:
		{
#ifdef FSV_LOGGING
			f_sprintf( szLogBuf,
				"RecDel: CO = %4.4X, DRN = %8.8X",
				(unsigned)uiContainer, (unsigned)uiDrn);
#endif

			if( RC_BAD( opRc = FlmRecordDelete( hDb,
				uiContainer,
				uiDrn,
				uiAutoTrans)))
			{
				goto OP_EXIT;
			}
			break;
		}

		case FCS_OP_RESERVE_NEXT_DRN:
		{
#ifdef FSV_LOGGING
			f_sprintf( szLogBuf,
				"ResDRN: CO = %4.4X, DRN = %8.8X",
				(unsigned)uiContainer, (unsigned)uiDrn);
#endif

			uiDrnRV = uiDrn;
			if( RC_BAD( opRc = FlmReserveNextDrn( hDb,
				uiContainer,
				&uiDrnRV)))
			{
				goto OP_EXIT;
			}

#ifdef FSV_LOGGING
			/*
			Need to change the log buffer after the operation so that
			it correctly represents the outcome of the operation.
			*/

			f_sprintf( szLogBuf,
				"RecAdd: CO = %4.4X, DRN = %8.8X",
				(unsigned)uiContainer, (unsigned)uiDrnRV);
#endif

			break;
		}

		case FCS_OP_KEY_RETRIEVE:
		{
#ifdef FSV_LOGGING
			f_sprintf( szLogBuf,
				"KeyRtrv: IX=%04X, CO=%04X, DRN=%08X",
				(unsigned)uiIndex, (unsigned)uiContainer, (unsigned)uiDrn);
#endif

			if (pSession->getClientVersion() >= FCS_VERSION_1_1_1)
			{
				if( RC_BAD( opRc = FlmKeyRetrieve( hDb,
					uiIndex,
					uiContainer,
					pRecord,
					uiDrn,
					uiFlags,
					&pRecordRV,
					&uiDrnRV)))
				{
					goto OP_EXIT;
				}
			}
			else
			{
				FLMUINT	uiKeyContainer = 0;

				if (pRecord)
				{
					uiKeyContainer = pRecord->getContainerID();
				}

				// Older clients sent index # in the container tag.

				if( RC_BAD( opRc = FlmKeyRetrieve( hDb,
					uiContainer,
					uiKeyContainer,
					pRecord,
					uiDrn,
					uiFlags,
					&pRecordRV,
					&uiDrnRV)))
				{
					goto OP_EXIT;
				}
			}
			break;
		}

		default:
		{
			opRc = RC_SET( FERR_NOT_IMPLEMENTED);
			goto OP_EXIT;
		}
	}
	
OP_EXIT:

	/*
	Send the server's response.
	*/

	if( RC_BAD( rc = pWire->sendOpcode( FCS_OPCLASS_RECORD, pWire->getOp())))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pWire->sendRc( opRc)))
	{
		goto Exit;
	}

	if( RC_OK( opRc))
	{
		if( pRecordRV)
		{
			if( RC_BAD( rc = pWire->sendRecord( WIRE_VALUE_RECORD, pRecordRV)))
			{
				goto Exit;
			}
		}

		if( uiDrnRV)
		{
			if( RC_BAD( rc = pWire->sendNumber( WIRE_VALUE_DRN, uiDrnRV)))
			{
				goto Exit;
			}
		}
	}

	if( RC_BAD( rc = pWire->sendTerminate()))
	{
		goto Exit;
	}

Exit:

#ifdef FSV_LOGGING
	fsvLogHandlerMessage( NULL, szLogBuf, opRc ? opRc : rc, FSV_LOG_EVENT);
#endif

	if( pRecordRV)
	{
		pRecordRV->Release();
	}

	return( rc);
}


/****************************************************************************
Desc:    Performs a database operation.
*****************************************************************************/
RCODE fsvOpClassDatabase(
	FSV_WIRE *		pWire)
{
	RCODE					rc = FERR_OK;
	RCODE					opRc = FERR_OK;
	FSV_SESN *			pSession;
	HFDB					hDb = HFDB_NULL;
	CREATE_OPTS			CreateOptsRV;
	FLMUINT				uiBlockCountRV;
	FLMUINT				uiBlocksExaminedRV;
	FLMUINT				uiBlockAddrRV;
	FLMUINT				uiTransIdRV;
	FLMUINT64			ui64NumValue1RV = 0;
	FLMUINT64			ui64NumValue2RV = 0;
	FLMUINT64			ui64NumValue3RV = 0;
	FLMBOOL				bBoolValueRV = FALSE;
	FLMUINT				uiItemIdRV = 0;
	char					szItemName[ 64];
	NODE *				pHTDRV = NULL;
	char					szPathRV[ F_PATH_MAX_SIZE];
	F_NameTable 		nameTable;
	FLMBOOL				bHaveCreateOptsVal = FALSE;
	FLMBOOL				bHavePathValue = FALSE;
	FLMBYTE *			pBinary;
	FLMUINT				uiBinSize;
#ifdef FSV_LOGGING
	char					szLogBuf[ FSV_LOG_BUFFER_SIZE];
#endif

#ifdef FSV_LOGGING
	szLogBuf[ 0] = 0;
#endif
	szItemName[ 0] = 0;

	if( (pSession = pWire->getSession()) == NULL)
	{
		opRc = RC_SET( FERR_BAD_HDL);
		goto OP_EXIT;
	}

	if( pWire->getOp() != FCS_OP_DATABASE_OPEN &&
		pWire->getOp() != FCS_OP_DATABASE_CREATE)
	{
		/*
		Get the database handle for all database operations other
		than open and create.
		*/

		if( (hDb = (HFDB)pWire->getFDB()) == HFDB_NULL)
		{
#ifdef FSV_LOGGING
			f_sprintf( szLogBuf, "Invalid database handle.");
#endif
			opRc = RC_SET( FERR_BAD_HDL);
			goto OP_EXIT;
		}
	}

	switch( pWire->getOp())
	{
		case FCS_OP_DATABASE_OPEN:
		{
#ifdef FSV_LOGGING
			f_sprintf( szLogBuf, "DBOpen");
#endif

			if( RC_BAD( opRc = pSession->OpenDatabase(
				pWire->getFilePath(),
				pWire->getFilePath3(),
				pWire->getFilePath2(),
				pWire->getFlags())))
			{
				goto OP_EXIT;
			}

#ifdef FSV_LOGGING
			f_sprintf( szLogBuf, "DBOpen");
#endif
			break;
		}

		case FCS_OP_DATABASE_CREATE:
		{
#ifdef FSV_LOGGING
			f_sprintf( szLogBuf, "DBCreate");
#endif
			CREATE_OPTS    createOpts;
			pWire->copyCreateOpts( &createOpts);

			if( RC_BAD( opRc = pSession->CreateDatabase(
				pWire->getFilePath(),
				pWire->getFilePath3(),
				pWire->getFilePath2(),
				pWire->getDictPath(),
				pWire->getDictBuffer(),
				&createOpts)))
			{
				goto OP_EXIT;
			}

#ifdef FSV_LOGGING
			f_sprintf( szLogBuf, "DBCreate");
#endif
			break;
		}

		case FCS_OP_DATABASE_CLOSE:
		{
#ifdef FSV_LOGGING
			f_sprintf( szLogBuf, "DBClose");
#endif

			if( RC_BAD( opRc = pSession->CloseDatabase()))
			{
				goto OP_EXIT;
			}
			break;
		}

		case FCS_OP_DB_REDUCE_SIZE:
		{
#ifdef FSV_LOGGING
			f_sprintf( szLogBuf,
				"DbReduce: BL = %8.8X", (unsigned)pWire->getCount());
#endif
			if( RC_BAD( opRc = FlmDbReduceSize(
				hDb,
				(FLMUINT)pWire->getCount(),
				&uiBlockCountRV)))
			{
				goto OP_EXIT;
			}
			break;
		}

		case FCS_OP_GET_ITEM_NAME:
		{
#ifdef FSV_LOGGING
			f_sprintf( szLogBuf,
				"GetItemName: ID = %4.4X", (unsigned)pWire->getItemId());
#endif

			if( RC_BAD( opRc = FlmGetItemName( hDb,
				pWire->getItemId(), sizeof( szItemName), szItemName)))
			{
				goto OP_EXIT;
			}
			break;
		}

		case FCS_OP_GET_NAME_TABLE:
		{
#ifdef FSV_LOGGING
			f_sprintf( szLogBuf, "GetNameTable");
#endif
			if( RC_BAD( rc = nameTable.setupFromDb( hDb)))
			{
				goto OP_EXIT;
			}
			break;
		}

		case FCS_OP_GET_COMMIT_CNT:
		{
#ifdef FSV_LOGGING
			f_sprintf( szLogBuf, "DbGetCommitCnt");
#endif
			if( RC_BAD( opRc = FlmDbGetCommitCnt(
				hDb,
				&uiBlockCountRV)))
			{
				goto OP_EXIT;
			}
			break;
		}

		case FCS_OP_GET_TRANS_ID:
		{
#ifdef FSV_LOGGING
			f_sprintf( szLogBuf, "DbGetTransId");
#endif
			if( RC_BAD( opRc = FlmDbGetTransId(
				hDb, &uiTransIdRV)))
			{
				goto OP_EXIT;
			}
			break;
		}

		case FCS_OP_DATABASE_GET_CONFIG:
		{
#ifdef FSV_LOGGING
			f_sprintf( szLogBuf,
				"DBGetConfig: TY = %8.8X", (unsigned)pWire->getType());
#endif
			switch( (eDbGetConfigType)pWire->getType())
			{
				case FDB_GET_VERSION:

					// Doing via create opts to maintain backward compatibility.

					f_memset( &CreateOptsRV, 0, sizeof( CreateOptsRV));
					if( RC_BAD( opRc = FlmDbGetConfig( hDb,
						FDB_GET_VERSION,
						(void *)&CreateOptsRV.uiVersionNum)))
					{
						goto OP_EXIT;
					}
					bHaveCreateOptsVal = TRUE;
					break;
				case FDB_GET_BLKSIZ:

					// Doing via create opts to maintain backward compatibility.

					f_memset( &CreateOptsRV, 0, sizeof( CreateOptsRV));
					if( RC_BAD( opRc = FlmDbGetConfig( hDb,
						FDB_GET_BLKSIZ,
						(void *)&CreateOptsRV.uiBlockSize)))
					{
						goto OP_EXIT;
					}
					bHaveCreateOptsVal = TRUE;
					break;
				case FDB_GET_DEFAULT_LANG:

					// Doing via create opts to maintain backward compatibility.

					f_memset( &CreateOptsRV, 0, sizeof( CreateOptsRV));
					if( RC_BAD( opRc = FlmDbGetConfig( hDb,
						FDB_GET_DEFAULT_LANG,
						(void *)&CreateOptsRV.uiDefaultLanguage)))
					{
						goto OP_EXIT;
					}
					bHaveCreateOptsVal = TRUE;
					break;

				case FDB_GET_TRANS_ID:
				case FDB_GET_RFL_FILE_NUM:
				case FDB_GET_RFL_HIGHEST_NU:
				case FDB_GET_LAST_BACKUP_TRANS_ID:
				case FDB_GET_BLOCKS_CHANGED_SINCE_BACKUP:
				case FDB_GET_FILE_EXTEND_SIZE:
				case FDB_GET_APP_DATA:
				{
					FLMUINT	uiTmpValue;

					if( RC_BAD( opRc = FlmDbGetConfig(
						hDb, (eDbGetConfigType)pWire->getType(), (void *)&uiTmpValue)))
					{
						goto OP_EXIT;
					}
					ui64NumValue1RV = (FLMUINT64)uiTmpValue;
					break;
				}
				case FDB_GET_RFL_FILE_SIZE_LIMITS:
				{
					FLMUINT	uiTmpValue1;
					FLMUINT	uiTmpValue2;

					if( RC_BAD( opRc = FlmDbGetConfig(
						hDb, FDB_GET_RFL_FILE_SIZE_LIMITS,
						(void *)&uiTmpValue1,
						(void *)&uiTmpValue2)))
					{
						goto OP_EXIT;
					}
					ui64NumValue1RV = (FLMUINT64)uiTmpValue1;
					ui64NumValue2RV = (FLMUINT64)uiTmpValue2;
					break;
				}

				case FDB_GET_RFL_KEEP_FLAG:
				case FDB_GET_AUTO_TURN_OFF_KEEP_RFL_FLAG:
				case FDB_GET_KEEP_ABORTED_TRANS_IN_RFL_FLAG:
				{
					if( RC_BAD( opRc = FlmDbGetConfig(
						hDb, (eDbGetConfigType)pWire->getType(), (void *)&bBoolValueRV)))
					{
						goto OP_EXIT;
					}
					break;
				}

				case FDB_GET_PATH:
				{
					if( RC_BAD( opRc = FlmDbGetConfig(
						hDb, FDB_GET_PATH,
						(void *)szPathRV)))
					{
						goto OP_EXIT;
					}
					bHavePathValue = TRUE;
					break;
				}

				case FDB_GET_CHECKPOINT_INFO:
				{
					CHECKPOINT_INFO	checkpointInfo;

					if( RC_BAD( opRc = FlmDbGetConfig(
						hDb, FDB_GET_CHECKPOINT_INFO,
						(void *)&checkpointInfo)))
					{
						goto OP_EXIT;
					}

					if( RC_BAD( opRc = fcsBuildCheckpointInfo( 
						&checkpointInfo, pWire->getPool(), &pHTDRV)))
					{
						goto OP_EXIT;
					}
					break;
				}

				case FDB_GET_LOCK_HOLDER:
				{
					LOCK_USER	lockUser;

					if( RC_BAD( opRc = FlmDbGetConfig(
						hDb, FDB_GET_LOCK_HOLDER,
						(void *)&lockUser)))
					{
						goto OP_EXIT;
					}

					if( RC_BAD( opRc = fcsBuildLockUser( 
						&lockUser, FALSE, pWire->getPool(), &pHTDRV)))
					{
						goto OP_EXIT;
					}
					break;
				}

				case FDB_GET_LOCK_WAITERS:
				{
					LOCK_USER *		pLockUser = NULL;

					if( RC_BAD( opRc = FlmDbGetConfig(
						hDb, FDB_GET_LOCK_WAITERS,
						(void *)&pLockUser)))
					{
						if( pLockUser)
						{
							FlmFreeMem( &pLockUser);
						}
						goto OP_EXIT;
					}

					if( RC_BAD( opRc = fcsBuildLockUser( 
						pLockUser, TRUE, pWire->getPool(), &pHTDRV)))
					{
						if( pLockUser)
						{
							FlmFreeMem( &pLockUser);
						}
						goto OP_EXIT;
					}

					if( pLockUser)
					{
						FlmFreeMem( &pLockUser);
					}
					break;
				}

				case FDB_GET_RFL_DIR:
				{
					if( RC_BAD( opRc = FlmDbGetConfig(
						hDb, FDB_GET_RFL_DIR,
						(void *)szPathRV)))
					{
						goto OP_EXIT;
					}
					bHavePathValue = TRUE;
					break;
				}

				case FDB_GET_SERIAL_NUMBER:
				{
					uiBinSize = F_SERIAL_NUM_SIZE;

					pBinary = (FLMBYTE *)GedPoolAlloc( pWire->getPool(), 
						uiBinSize);
					
					if( !pBinary) 
					{
						opRc = RC_SET( FERR_MEM);
						goto OP_EXIT;
					}

					if( RC_BAD( opRc = FlmDbGetConfig(
						hDb, FDB_GET_SERIAL_NUMBER,
						(void *)pBinary)))
					{
						goto OP_EXIT;
					}
					break;
				}

				case FDB_GET_SIZES:
				{
					if( RC_BAD( opRc = FlmDbGetConfig(
						hDb, FDB_GET_SIZES,
						(void *)&ui64NumValue1RV,
						(void *)&ui64NumValue2RV,
						(void *)&ui64NumValue3RV)))
					{
						goto OP_EXIT;
					}
					break;
				}

				default:
				{
					opRc = RC_SET( FERR_NOT_IMPLEMENTED);
					goto OP_EXIT;
				}
			}

			break;
		}

		case FCS_OP_DATABASE_CONFIG:
		{
#ifdef FSV_LOGGING
			f_sprintf( szLogBuf,
				"DBConfig: TY = %8.8X", (unsigned)pWire->getType());
#endif
			switch( (eDbConfigType)pWire->getType())
			{
				case FDB_SET_APP_VERSION:
				case FDB_RFL_KEEP_FILES:
				case FDB_RFL_ROLL_TO_NEXT_FILE:
				case FDB_KEEP_ABORTED_TRANS_IN_RFL:
				case FDB_AUTO_TURN_OFF_KEEP_RFL:
				case FDB_SET_APP_DATA:
					if( RC_BAD( opRc = FlmDbConfig( hDb, (eDbConfigType)pWire->getType(),
						(void *)((FLMUINT)pWire->getNumber2()),
						(void *)((FLMUINT)pWire->getNumber3()))))
					{
						goto OP_EXIT;
					}
					break;
				case FDB_RFL_FILE_LIMITS:
				case FDB_FILE_EXTEND_SIZE:
					if( RC_BAD( opRc = FlmDbConfig( hDb, (eDbConfigType)pWire->getType(),
						(void *)((FLMUINT)pWire->getNumber1()),
						(void *)((FLMUINT)pWire->getNumber2()))))
					{
						goto OP_EXIT;
					}
					break;
				
				case FDB_RFL_DIR:
				{
					char *		pszPath;
					POOL *		pPool = pWire->getPool();
					void *		pvMark = GedPoolMark( pPool);

					if( RC_BAD( rc = fcsConvertUnicodeToNative( pPool,
						pWire->getFilePath(), &pszPath)))
					{
						goto Exit;
					}

					if( RC_BAD( opRc = FlmDbConfig( hDb, (eDbConfigType)pWire->getType(),
						(void *)pszPath, (void *)((FLMUINT)pWire->getNumber3()))))
					{
						goto OP_EXIT;
					}

					GedPoolReset( pPool, pvMark);
					break;
				}

				default:
					opRc = RC_SET( FERR_NOT_IMPLEMENTED);
					goto OP_EXIT;
			}

			break;
		}

		case FCS_OP_DATABASE_LOCK:
		{
#ifdef FSV_LOGGING
			f_sprintf( szLogBuf, "DbLock");
#endif
			if( RC_BAD( opRc = FlmDbLock(
				hDb,
				(FLOCK_TYPE)(FLMUINT)pWire->getNumber1(),
				(FLMINT)pWire->getSignedValue(),
				(FLMUINT)pWire->getFlags())))
			{
				goto OP_EXIT;
			}
			break;
		}

		case FCS_OP_DATABASE_UNLOCK:
		{
#ifdef FSV_LOGGING
			f_sprintf( szLogBuf, "DbUnlock");
#endif
			if( RC_BAD( opRc = FlmDbUnlock( hDb)))
			{
				goto OP_EXIT;
			}
			break;
		}

		case FCS_OP_DATABASE_GET_BLOCK:
		{
#ifdef FSV_LOGGING
			f_sprintf( szLogBuf, "DbGetBlock");
#endif
			uiBlockCountRV = (FLMUINT)pWire->getCount();
			if( RC_BAD( opRc = fsvDbGetBlocks( hDb, pWire->getAddress(),
				pWire->getTransId(),
				&uiBlockCountRV, &uiBlocksExaminedRV, &uiBlockAddrRV, 
				pWire->getFlags(), pWire->getPool(), &pBinary, &uiBinSize)))
			{
				goto OP_EXIT;
			}
			break;
		}

		case FCS_OP_DATABASE_CHECKPOINT:
		{
#ifdef FSV_LOGGING
			f_sprintf( szLogBuf, "DbCheckpoint");
#endif
			if( RC_BAD( opRc = FlmDbCheckpoint(
				hDb,
				pWire->getFlags() /* timeout */)))
			{
				goto OP_EXIT;
			}
			break;
		}

		case FCS_OP_DB_SET_BACKUP_FLAG:
		{
			FLMBOOL	bNewState = pWire->getBoolean();

#ifdef FSV_LOGGING
			f_sprintf( szLogBuf, "SetBackupFlag");
#endif
			if( !IsInCSMode( hDb))
			{
				FDB *		pDb = (FDB *)hDb;

				f_mutexLock( gv_FlmSysData.hShareMutex);
				if( pDb->pFile->bBackupActive && bNewState)
				{
					f_mutexUnlock( gv_FlmSysData.hShareMutex);
					opRc = RC_SET( FERR_BACKUP_ACTIVE);
					goto OP_EXIT;
				}
				pDb->pFile->bBackupActive = bNewState;
				f_mutexUnlock( gv_FlmSysData.hShareMutex);
			}
			else
			{
				if( RC_BAD( opRc = fcsSetBackupActiveFlag( 
					hDb, bNewState)))
				{
					goto OP_EXIT;
				}
			}

			break;
		}

		default:
		{
			opRc = RC_SET( FERR_NOT_IMPLEMENTED);
			goto OP_EXIT;
		}
	}
	
OP_EXIT:

	/*
	Send the server's response.
	*/

	if( RC_BAD( rc = pWire->sendOpcode(
		FCS_OPCLASS_DATABASE, pWire->getOp())))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pWire->sendRc( opRc)))
	{
		goto Exit;
	}

	switch( pWire->getOp())
	{
		case FCS_OP_DB_REDUCE_SIZE:
		case FCS_OP_GET_COMMIT_CNT:
		{
			/*
			Return a count
			*/

			if( RC_BAD( rc = pWire->sendNumber( WIRE_VALUE_COUNT,
				uiBlockCountRV)))
			{
				goto Exit;
			}
			break;
		}

		case FCS_OP_GET_NAME_TABLE:
		{
			/*
			Return the name table.
			*/

			if( RC_OK( opRc))
			{
				if( RC_BAD( rc = pWire->sendNameTable(	
					WIRE_VALUE_NAME_TABLE, &nameTable)))
				{
					goto Exit;
				}
			}
			break;
		}

		case FCS_OP_GET_ITEM_NAME:
		{
			FLMUNICODE *	puzItemNameRV;

			if( RC_OK( opRc))
			{
				if( szItemName[ 0])
				{
					if( RC_BAD( rc = fcsConvertNativeToUnicode( 
						pWire->getPool(), szItemName, &puzItemNameRV)))
					{
						goto Exit;
					}

					if( RC_BAD( rc = pWire->sendString( 
						WIRE_VALUE_ITEM_NAME, puzItemNameRV)))
					{
						goto Exit;
					}
				}
			}
			break;
		}

		case FCS_OP_GET_ITEM_ID:
		{
			if( uiItemIdRV)
			{
				if( RC_BAD( rc = pWire->sendNumber( 
					WIRE_VALUE_ITEM_ID, uiItemIdRV)))
				{
					goto Exit;
				}
			}
			break;
		}

		case FCS_OP_GET_TRANS_ID:
		{
			/*
			Return the transaction id for the database.
			*/

			if( RC_BAD( rc = pWire->sendNumber(
				WIRE_VALUE_TRANSACTION_ID, uiTransIdRV)))
			{
				goto Exit;
			}
			break;
		}

		case FCS_OP_DATABASE_GET_BLOCK:
		{
			/*
			Return the requested block
			*/

			if( RC_BAD( rc = pWire->sendNumber( WIRE_VALUE_COUNT,
				uiBlockCountRV)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = pWire->sendNumber( WIRE_VALUE_NUMBER2,
				uiBlocksExaminedRV)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = pWire->sendNumber( WIRE_VALUE_ADDRESS,
				uiBlockAddrRV)))
			{
				goto Exit;
			}

			if( uiBlockCountRV)
			{
				if( RC_BAD( rc = pWire->sendBinary(
					WIRE_VALUE_BLOCK, pBinary, uiBinSize)))
				{
					goto Exit;
				}
			}

			break;
		}

		case FCS_OP_DATABASE_GET_CONFIG:
		{
			switch( pWire->getType())
			{
				case FDB_GET_SERIAL_NUMBER:
					if( RC_BAD( rc = pWire->sendBinary(
						WIRE_VALUE_SERIAL_NUM, pBinary, uiBinSize)))
					{
						goto Exit;
					}
					break;
				default:
					break;
			}
			break;
		}
	}

	if( bHaveCreateOptsVal)
	{
		if( RC_BAD( rc = pWire->sendCreateOpts( 
			WIRE_VALUE_CREATE_OPTS, &CreateOptsRV)))
		{
			goto Exit;
		}
	}

	if( ui64NumValue1RV)
	{
		if( RC_BAD( rc = pWire->sendNumber( 
			WIRE_VALUE_NUMBER1, ui64NumValue1RV)))
		{
			goto Exit;
		}
	}

	if( ui64NumValue2RV)
	{
		if( RC_BAD( rc = pWire->sendNumber( 
			WIRE_VALUE_NUMBER2, ui64NumValue2RV)))
		{
			goto Exit;
		}
	}

	if( ui64NumValue3RV)
	{
		if( RC_BAD( rc = pWire->sendNumber( 
			WIRE_VALUE_NUMBER3, ui64NumValue3RV)))
		{
			goto Exit;
		}
	}

	if( bBoolValueRV)
	{
		if( RC_BAD( rc = pWire->sendNumber( 
			WIRE_VALUE_BOOLEAN, bBoolValueRV)))
		{
			goto Exit;
		}
	}

	if( bHavePathValue)
	{
		FLMUNICODE *		puzPath;

		if( RC_BAD( rc = fcsConvertNativeToUnicode( pWire->getPool(),
			szPathRV, &puzPath)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = pWire->sendString( 
			WIRE_VALUE_FILE_PATH, puzPath)))
		{
			goto Exit;
		}
	}

	if( pHTDRV)
	{
		if( RC_BAD( rc = pWire->sendHTD( WIRE_VALUE_HTD, pHTDRV)))
		{
			goto Exit;
		}
	}

	if( RC_BAD( rc = pWire->sendTerminate()))
	{
		goto Exit;
	}

Exit:

#ifdef FSV_LOGGING
	fsvLogHandlerMessage( NULL, szLogBuf, opRc ? opRc : rc, FSV_LOG_EVENT);
#endif

	return( rc);
}


/****************************************************************************
Desc:    Performs an iterator (cursor) operation
*****************************************************************************/
RCODE fsvOpClassIterator(
	FSV_WIRE *		pWire)
{
	RCODE			rc = FERR_OK;
	RCODE			opRc = FERR_OK;
	FSV_SESN *	pSession = NULL;
	HFCURSOR		hIterator = HFCURSOR_NULL;
	FLMBOOL		bDoDrnOp = FALSE;
	FlmRecord *	pRecordRV = NULL;
	FlmRecord *	pTmpRecord = NULL;
	FLMUINT		uiIteratorIdRV = FCS_INVALID_ID;
	FLMUINT		uiCountRV = 0;
	FLMUINT		uiDrnRV = 0;
	FLMBOOL		bFlag = FALSE;
#ifdef FSV_LOGGING
	char			szLogBuf[ FSV_LOG_BUFFER_SIZE];
#endif

#ifdef FSV_LOGGING
	szLogBuf[ 0] = 0;
#endif

	/*
	Get a pointer to the session object.
	*/

	if( (pSession = pWire->getSession()) == NULL)
	{
		opRc = RC_SET( FERR_BAD_HDL);
		goto OP_EXIT;
	}

	/*
	Get the iterator handle.
	*/
	
	if( (hIterator = pWire->getIteratorHandle()) == HFDB_NULL)
	{
		if( pWire->getOp() != FCS_OP_ITERATOR_INIT)
		{
#ifdef FSV_LOGGING
			f_sprintf( szLogBuf, "Invalid iterator handle.");
#endif
			opRc = RC_SET( FERR_BAD_HDL);
			goto OP_EXIT;
		}
	}

	/*
	Examine the wire flags for the operation.
	*/

	bDoDrnOp = (FLMBOOL)((pWire->getFlags() & FCS_ITERATOR_DRN_FLAG)
					? (FLMBOOL)TRUE
					: (FLMBOOL)FALSE);

	/*
	Perform the requested operation.
	*/

	switch( pWire->getOp())
	{
		case FCS_OP_ITERATOR_INIT:
		{
#ifdef FSV_LOGGING
			f_sprintf( szLogBuf, "IteratorInit");
#endif

			/*
			Build the query.
			*/

			if( RC_BAD( opRc = fsvIteratorParse( pWire, pWire->getPool())))
			{
				goto OP_EXIT;
			}
			uiIteratorIdRV = pWire->getIteratorId();

#ifdef FSV_LOGGING
			f_sprintf( szLogBuf, "IteratorInit: ID = %8.8X",
				(unsigned)uiIteratorIdRV);
#endif
			break;
		}

		case FCS_OP_ITERATOR_FREE:
		{
#ifdef FSV_LOGGING
			f_sprintf( szLogBuf, "IteratorFree: ID = %8.8X",
				(unsigned)pWire->getIteratorId());
#endif
			/*
			Free the iterator.
			*/

			if( RC_BAD( opRc = pSession->FreeIterator( pWire->getIteratorId())))
			{
				goto OP_EXIT;
			}

			break;
		}

		case FCS_OP_ITERATOR_FIRST:
		{
#ifdef FSV_LOGGING
			f_sprintf( szLogBuf, "IteratorFirst: ID = %8.8X",
				(unsigned)pWire->getIteratorId());
#endif

			/*
			Retrieve the first record (or DRN) in the result set.
			*/

			if( bDoDrnOp)
			{
				if( RC_BAD( opRc = FlmCursorFirstDRN( hIterator, &uiDrnRV)))
				{
					goto OP_EXIT;
				}
			}
			else
			{
				if( RC_BAD( opRc = FlmCursorFirst( hIterator, &pRecordRV)))
				{
					goto OP_EXIT;
				}
			}
			break;
		}
		
		case FCS_OP_ITERATOR_LAST:
		{
#ifdef FSV_LOGGING
			f_sprintf( szLogBuf, "IteratorLast: ID = %8.8X",
				(unsigned)pWire->getIteratorId());
#endif

			/*
			Retrieve the last record (or DRN) in the result set.
			*/

			if( bDoDrnOp)
			{
				if( RC_BAD( opRc = FlmCursorLastDRN( hIterator, &uiDrnRV)))
				{
					goto OP_EXIT;
				}
			}
			else
			{
				if( RC_BAD( opRc = FlmCursorLast( hIterator, &pRecordRV)))
				{
					goto OP_EXIT;
				}
			}
			break;
		}

		case FCS_OP_ITERATOR_NEXT:
		{
#ifdef FSV_LOGGING
			f_sprintf( szLogBuf, "IteratorNext: ID = %8.8X",
				(unsigned)pWire->getIteratorId());
#endif

			/*
			Retrieve the next record (or DRN) in the result set.
			*/

			if( bDoDrnOp)
			{
				if( RC_BAD( opRc = FlmCursorNextDRN( hIterator, &uiDrnRV)))
				{
					goto OP_EXIT;
				}
			}
			else
			{
				if( RC_BAD( opRc = FlmCursorNext( hIterator, &pRecordRV)))
				{
					goto OP_EXIT;
				}
			}
			break;
		}

		case FCS_OP_ITERATOR_PREV:
		{
#ifdef FSV_LOGGING
			f_sprintf( szLogBuf, "IteratorPrev: ID = %8.8X",
				(unsigned)pWire->getIteratorId());
#endif

			/*
			Retrieve the previous record (or DRN) in the result set.
			*/

			if( bDoDrnOp)
			{
				if( RC_BAD( opRc = FlmCursorPrevDRN( hIterator, &uiDrnRV)))
				{
					goto OP_EXIT;
				}
			}
			else
			{
				if( RC_BAD( opRc = FlmCursorPrev( hIterator, &pRecordRV)))
				{
					goto OP_EXIT;
				}
			}
			break;
		}

		case FCS_OP_ITERATOR_COUNT:
		{
#ifdef FSV_LOGGING
			f_sprintf( szLogBuf, "IteratorCount: ID = %8.8X",
				(unsigned)pWire->getIteratorId());
#endif

			/*
			Count the number of records in the result set.
			*/

			if( RC_BAD( opRc = FlmCursorRecCount( hIterator, &uiCountRV)))
			{
				goto OP_EXIT;
			}
			break;
		}

		case FCS_OP_ITERATOR_TEST_REC:
		{
			if ((pTmpRecord = pWire->getRecord()) != NULL)
			{
				pTmpRecord->AddRef();

#ifdef FSV_LOGGING
				f_sprintf( szLogBuf, "IteratorTestRec: ID = %8.8X",
					(unsigned)pWire->getIteratorId());
#endif
				if( RC_BAD( opRc = FlmCursorTestRec( hIterator, pTmpRecord, &bFlag)))
				{
					goto OP_EXIT;
				}
				pTmpRecord->Release();
				pTmpRecord = NULL;
			}
			else
			{
#ifdef FSV_LOGGING
				f_sprintf( szLogBuf, "IteratorTestDRN: ID = %8.8X",
					(unsigned)pWire->getIteratorId());
#endif
				if( RC_BAD( opRc = FlmCursorTestDRN( hIterator, pWire->getDrn(),
												&bFlag)))
				{
					goto OP_EXIT;
				}
			}
			break;
		}

		default:
		{
			opRc = RC_SET( FERR_NOT_IMPLEMENTED);
			goto OP_EXIT;
		}
	}
	
OP_EXIT:

	/*
	Send the server's response.
	*/

	if( RC_BAD( rc = pWire->sendOpcode( FCS_OPCLASS_ITERATOR, pWire->getOp())))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pWire->sendRc( opRc)))
	{
		goto Exit;
	}

	if( RC_OK( opRc))
	{
		if( pRecordRV)
		{
			/*
			Send the retrieved record.
			*/

			if( RC_BAD( rc = pWire->sendRecord( WIRE_VALUE_RECORD, pRecordRV)))
			{
				goto Exit;
			}
		}

		if( uiDrnRV)
		{
			/*
			Send the record's DRN.
			*/

			if( RC_BAD( rc = pWire->sendNumber( WIRE_VALUE_DRN, uiDrnRV)))
			{
				goto Exit;
			}
		}

		if( uiCountRV)
		{
			/*
			Send the record count.
			*/

			if( RC_BAD( rc = pWire->sendNumber( 
				WIRE_VALUE_RECORD_COUNT, uiCountRV)))
			{
				goto Exit;
			}
		}

		if( uiIteratorIdRV != FCS_INVALID_ID)
		{
			/*
			Send the iterator's ID.
			*/

			if( RC_BAD( rc = pWire->sendNumber( 
				WIRE_VALUE_ITERATOR_ID, uiIteratorIdRV)))
			{
				goto Exit;
			}
		}

		if (bFlag)
		{
			if( RC_BAD( rc = pWire->sendNumber( WIRE_VALUE_BOOLEAN, bFlag)))
			{
				goto Exit;
			}
		}
	}

	if( RC_BAD( rc = pWire->sendTerminate()))
	{
		goto Exit;
	}

Exit:

#ifdef FSV_LOGGING
	fsvLogHandlerMessage( NULL, szLogBuf, opRc ? opRc : rc, FSV_LOG_EVENT);
#endif

	if( pRecordRV)
	{
		pRecordRV->Release();
	}

	if( pTmpRecord)
	{
		pTmpRecord->Release();
		pTmpRecord = NULL;
	}

	return( rc);
}

/****************************************************************************
Desc:    Performs a transaction operation
*****************************************************************************/
RCODE fsvOpClassTransaction(
	FSV_WIRE *		pWire)
{
	RCODE			rc = FERR_OK;
	RCODE			opRc = FERR_OK;
	FSV_SESN *	pSession;
	HFDB			hDb;
	FLMUINT		uiTransTypeRV;
#ifdef FSV_LOGGING
	char			szLogBuf[ FSV_LOG_BUFFER_SIZE];
#endif
	FLMBYTE *	pBlock = NULL;
	FLMUINT		uiBlockSize = 0;
	FLMUINT		uiFlmTransFlags = 0;

#ifdef FSV_LOGGING
	szLogBuf[ 0] = 0;
#endif

	/*
	Get a pointer to the session object.
	*/

	if( (pSession = pWire->getSession()) == NULL)
	{
		opRc = RC_SET( FERR_BAD_HDL);
		goto OP_EXIT;
	}

	/*
	Get a handle to the database in case this is a
	database transaction operation
	*/

	hDb = (HFDB)pWire->getFDB();

	/*
	Perform the requested operation.
	*/
	
	switch( pWire->getOp())
	{
		case FCS_OP_TRANSACTION_BEGIN:
		{
#ifdef FSV_LOGGING
			f_sprintf( szLogBuf,
				"TransBegin: TT = %8.8X",	(unsigned)pWire->getTransType());
#endif
			/*
			Start a database transaction.
			*/

			if( pWire->getFlags() & FCS_TRANS_FLAG_GET_HEADER)
			{
				uiBlockSize = 2048;
				if( (pBlock = (FLMBYTE *)GedPoolAlloc( 
					pWire->getPool(), uiBlockSize)) == NULL)
				{
					rc = RC_SET( FERR_MEM);
					goto OP_EXIT;
				}
			}
			
			if( pWire->getFlags() & FCS_TRANS_FLAG_DONT_KILL)
			{
				uiFlmTransFlags |= FLM_DONT_KILL_TRANS;
			}

			if( pWire->getFlags() & FCS_TRANS_FLAG_DONT_POISON)
			{
				uiFlmTransFlags |= FLM_DONT_POISON_CACHE;
			}

			if( RC_BAD( opRc = FlmDbTransBegin( hDb,
				pWire->getTransType() | uiFlmTransFlags, 
				pWire->getMaxLockWait(), pBlock)))
			{
				goto OP_EXIT;
			}
			break;
		}

		case FCS_OP_TRANSACTION_COMMIT:
		{
#ifdef FSV_LOGGING
			f_sprintf( szLogBuf, "TransCommit");
#endif
			/*
			Commit a database transaction.
			*/

			if( RC_BAD( opRc = FlmDbTransCommit( hDb)))
			{
				goto OP_EXIT;
			}
			break;
		}

		case FCS_OP_TRANSACTION_COMMIT_EX:
		{
#ifdef FSV_LOGGING
			f_sprintf( szLogBuf, "TransCommitEx");
#endif
			/*
			Commit a database transaction.
			*/

			if( RC_BAD( opRc = fsvDbTransCommitEx( hDb, pWire)))
			{
				goto OP_EXIT;
			}
			break;
		}

		case FCS_OP_TRANSACTION_ABORT:
		{
#ifdef FSV_LOGGING
			f_sprintf( szLogBuf, "TransAbort");
#endif
			/*
			Abort a database transaction.
			*/

			if( RC_BAD( opRc = FlmDbTransAbort( hDb)))
			{
				goto OP_EXIT;
			}
			break;
		}

		case FCS_OP_TRANSACTION_GET_TYPE:
		{
#ifdef FSV_LOGGING
			f_sprintf( szLogBuf, "GetTransType");
#endif
			/*
			Get the database transaction type.
			*/

			if( RC_BAD( opRc = FlmDbGetTransType( hDb, &uiTransTypeRV)))
			{
				goto OP_EXIT;
			}
			break;
		}
		
		default:
		{
			opRc = RC_SET( FERR_NOT_IMPLEMENTED);
			goto OP_EXIT;
		}
	}

OP_EXIT:

	/*
	Send the server's response.
	*/

	if( RC_BAD( rc = pWire->sendOpcode(
		FCS_OPCLASS_TRANS, pWire->getOp())))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pWire->sendRc( opRc)))
	{
		goto Exit;
	}

	if( pBlock)
	{
		if( RC_BAD( rc = pWire->sendBinary(
			WIRE_VALUE_BLOCK, pBlock, uiBlockSize)))
		{
			goto Exit;
		}
	}

	if( RC_OK( opRc))
	{
		switch( pWire->getOp())
		{
			case FCS_OP_TRANSACTION_GET_TYPE:
			{
				if( RC_BAD( rc = pWire->sendNumber(
					WIRE_VALUE_TRANSACTION_TYPE, uiTransTypeRV)))
				{
					goto Exit;
				}
				break;
			}
		}
	}

	if( RC_BAD( rc = pWire->sendTerminate()))
	{
		goto Exit;
	}

Exit:

#ifdef FSV_LOGGING
	fsvLogHandlerMessage( NULL, szLogBuf, opRc ? opRc : rc, FSV_LOG_EVENT);
#endif

	return( rc);
}


/****************************************************************************
Desc:    Performs a maintenance operation.
*****************************************************************************/
RCODE fsvOpClassMaintenance(
	FSV_WIRE *		pWire)
{
	FSV_SESN *		pSession;
	HFDB				hDb;
	POOL				pool;
#ifdef FSV_LOGGING
	char				szLogBuf[ FSV_LOG_BUFFER_SIZE];
#endif
	RCODE				opRc = FERR_OK;
	RCODE				rc = FERR_OK;

	/*
	Initialize a temporary pool.
	*/
	
	GedPoolInit( &pool, 1024);
	
	/*
	Initialize local variables.
	*/
	
#ifdef FSV_LOGGING
	szLogBuf[ 0] = '\0';
#endif

	/*
	Service the request.
	*/

	if( (pSession = pWire->getSession()) == NULL)
	{
		opRc = RC_SET( FERR_BAD_HDL);
		goto OP_EXIT;
	}

	if( (hDb = (HFDB)pWire->getFDB()) == HFDB_NULL)
	{
#ifdef FSV_LOGGING
		f_sprintf( szLogBuf,
			"Invalid database handle.");
#endif
		opRc = RC_SET( FERR_BAD_HDL);
		goto OP_EXIT;
	}

	switch( pWire->getOp())
	{
		case FCS_OP_CHECK:
		{
#ifdef FSV_LOGGING
			f_sprintf( szLogBuf, "Check");
#endif

			if( RC_BAD( opRc = FlmDbCheck( hDb, NULL, NULL, NULL,
				pWire->getFlags(), &pool, NULL, NULL, 0)))
			{
				goto OP_EXIT;
			}
			
			break;
		}

		default:
		{
			opRc = RC_SET( FERR_NOT_IMPLEMENTED);
			goto OP_EXIT;
		}
	}
	
OP_EXIT:

	/*
	Send the server's response.
	*/

	if( RC_BAD( rc = pWire->sendOpcode(
		FCS_OPCLASS_MAINTENANCE, pWire->getOp())))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pWire->sendRc( opRc)))
	{
		goto Exit;
	}

	if( RC_OK( opRc))
	{
		switch( pWire->getOp())
		{
			case FCS_OP_CHECK:
			{
				break;
			}
		}
	}

	if( RC_BAD( rc = pWire->sendTerminate()))
	{
		goto Exit;
	}

Exit:

#ifdef FSV_LOGGING
	fsvLogHandlerMessage( NULL, szLogBuf, opRc ? opRc : rc, FSV_LOG_EVENT);
#endif

	return( rc);
}


/****************************************************************************
Desc:    Performs an index operation
*****************************************************************************/
RCODE fsvOpClassIndex(
	FSV_WIRE *		pWire)
{
	HFDB				hDb = HFDB_NULL;
	FLMUINT			uiIndex;
	FINDEX_STATUS	indexStatus;
	POOL *			pTmpPool = pWire->getPool();
	RCODE				opRc = FERR_OK;
	RCODE				rc = FERR_OK;

	/*
	Get the database handle.  This is needed by all of the
	index operations.
	*/
	
	if( (hDb = (HFDB)pWire->getFDB()) == HFDB_NULL)
	{
		opRc = RC_SET( FERR_BAD_HDL);
		goto OP_EXIT;
	}

	/*
	Initialize local variables.
	*/

	uiIndex = pWire->getIndexId();

	/*
	Service the request.
	*/

	switch( pWire->getOp())
	{
		case FCS_OP_INDEX_GET_STATUS:
		{
			if( RC_BAD( opRc = FlmIndexStatus( hDb, uiIndex, &indexStatus)))
			{
				goto OP_EXIT;
			}
			break;
		}

		case FCS_OP_INDEX_GET_NEXT:
		{
			if( RC_BAD( opRc = FlmIndexGetNext( hDb, &uiIndex)))
			{
				goto OP_EXIT;
			}
			break;
		}

		case FCS_OP_INDEX_SUSPEND:
		{
			if( RC_BAD( opRc = FlmIndexSuspend( hDb, uiIndex)))
			{
				goto OP_EXIT;
			}
			break;
		}

		case FCS_OP_INDEX_RESUME:
		{
			if( RC_BAD( opRc = FlmIndexResume( hDb, uiIndex)))
			{
				goto OP_EXIT;
			}
			break;
		}

		default:
		{
			opRc = RC_SET( FERR_NOT_IMPLEMENTED);
			goto OP_EXIT;
		}
	}

OP_EXIT:

	/*
	Send the server's response.
	*/

	if( RC_BAD( rc = pWire->sendOpcode( FCS_OPCLASS_INDEX, pWire->getOp())))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pWire->sendRc( opRc)))
	{
		goto Exit;
	}

	if( RC_OK( opRc))
	{
		switch( pWire->getOp())
		{
			case FCS_OP_INDEX_GET_STATUS:
			{
				NODE *		pStatusTree;

				if( RC_BAD( fcsBuildIndexStatus( &indexStatus, 
					pTmpPool, &pStatusTree)))
				{
					goto Exit;
				}

				if( RC_BAD( rc = pWire->sendHTD( WIRE_VALUE_HTD, pStatusTree)))
				{
					goto Exit;
				}
				break;
			}

			case FCS_OP_INDEX_GET_NEXT:
			{
				if( RC_BAD( rc = pWire->sendNumber(
					WIRE_VALUE_INDEX_ID, uiIndex)))
				{
					goto Exit;
				}
				break;
			}
		}
	}

	if( RC_BAD( rc = pWire->sendTerminate()))
	{
		goto Exit;
	}

Exit:

	return( rc);
}


/****************************************************************************
Desc:    Performs a misc. operation
*****************************************************************************/
RCODE fsvOpClassMisc(
	FSV_WIRE *		pWire)
{
	FLMBYTE		ucSerialNum[ F_SERIAL_NUM_SIZE];
	RCODE			opRc = FERR_OK;
	RCODE			rc = FERR_OK;

	/*
	Service the request.
	*/

	switch( pWire->getOp())
	{
		case FCS_OP_CREATE_SERIAL_NUM:
		{
			if( RC_BAD( opRc = f_createSerialNumber( ucSerialNum)))
			{
				goto OP_EXIT;
			}
			break;
		}

		default:
		{
			opRc = RC_SET( FERR_NOT_IMPLEMENTED);
			goto OP_EXIT;
		}
	}


OP_EXIT:

	/*
	Send the server's response.
	*/

	if( RC_BAD( rc = pWire->sendOpcode( FCS_OPCLASS_MISC, pWire->getOp())))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pWire->sendRc( opRc)))
	{
		goto Exit;
	}

	if( RC_OK( opRc))
	{
		if( pWire->getOp() == FCS_OP_CREATE_SERIAL_NUM)
		{
			if( RC_BAD( rc = pWire->sendBinary(
				WIRE_VALUE_SERIAL_NUM, 
				ucSerialNum, F_SERIAL_NUM_SIZE)))
			{
				goto Exit;
			}
		}
		else
		{
			flmAssert( rc == FERR_NOT_IMPLEMENTED);
		}
	}

	if( RC_BAD( rc = pWire->sendTerminate()))
	{
		goto Exit;
	}

Exit:

	return( rc);
}


/****************************************************************************
Desc:    Configures an iterator based on from, where, select, and config
			clauses provided by the client.
*****************************************************************************/
FSTATIC RCODE fsvIteratorParse(
	FSV_WIRE *	pWire,
	POOL *		pPool)
{
	RCODE		rc = FERR_OK;

	/*
	Parse the "from" clause.  This contains record source information.
	*/

	if( pWire->getIteratorFrom())
	{
		if( RC_BAD( rc = fsvIteratorFromParse( pWire, pPool)))
		{
			goto Exit;
		}
	}
	if (pWire->getIteratorHandle() == HFCURSOR_NULL)
	{
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

	/*
	Parse the "where" clause.  This contains the criteria.
	*/
	
	if( pWire->getIteratorWhere())
	{
		if( RC_BAD( rc = fsvIteratorWhereParse( pWire, pPool)))
		{
			goto Exit;
		}
	}

	/*
	Parse the "select" clause.  This contains customized view information.
	*/

	if( pWire->getIteratorSelect())
	{
		if( RC_BAD( rc = fsvIteratorSelectParse( pWire, pPool)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}
	

/****************************************************************************
Desc:    Adds selection criteria to an iterator.
*****************************************************************************/
FSTATIC RCODE fsvIteratorWhereParse(
	FSV_WIRE *	pWire,
	POOL *		pPool)
{
	HFCURSOR		hIterator = pWire->getIteratorHandle();
	NODE *		pWhere = pWire->getIteratorWhere();
	NODE *		pCurNode;
	NODE *		pTmpNode;
	void *		pPoolMark;
	FLMUINT		uiTag;
	RCODE			rc = FERR_OK;

	/*
	If no "where" clause, jump to exit.
	*/

	if( !pWhere)
	{
		goto Exit;
	}

	/*
	Process each component of the "where" clause.
	*/

	pCurNode = GedChild( pWhere);
	while( pCurNode)
	{
		uiTag = GedTagNum( pCurNode);
		switch( uiTag)
		{
			case FCS_ITERATOR_MODE:
			{
				FLMUINT		uiFlags = 0;

				/*
				Set the iterator's mode flags
				*/

				if( RC_BAD( rc = GedGetUINT( pCurNode, &uiFlags)))
				{
					goto Exit;
				}

				if( RC_BAD( rc = FlmCursorSetMode( hIterator, uiFlags)))
				{
					goto Exit;
				}
				break;
			}

			/*
			Add an attribute to the criteria.
			*/

			case FCS_ITERATOR_ATTRIBUTE:
			{
				FLMUINT		uiAttrId;

				/*
				Get the attribute ID.
				*/
				
				if( RC_BAD( rc = GedGetUINT( pCurNode, &uiAttrId)))
				{
					goto Exit;
				}

				/*
				Add the attribute.
				*/

				if( uiTag == FCS_ITERATOR_ATTRIBUTE)
				{
					if( RC_BAD( rc = FlmCursorAddField( hIterator,
						uiAttrId, 0)))
					{
						goto Exit;
					}
				}
				else
				{
					/*
					Sanity check.
					*/

					flmAssert( 0);
				}
				break;
			}

			/*
			Add an attribute path to the criteria.
			*/

			case FCS_ITERATOR_ATTRIBUTE_PATH:
			{
				FLMUINT		puiPath[ FCS_ITERATOR_MAX_PATH + 1];
				FLMUINT		uiAttrId;
				FLMUINT		uiPathPos = 0;
				FLMUINT		uiStartLevel;

				if( (pTmpNode = GedFind( GED_TREE, pCurNode,
					FCS_ITERATOR_ATTRIBUTE, 1)) != NULL)
				{
					/*
					Build the attribute path.
					*/

					uiStartLevel = GedNodeLevel( pTmpNode);
					while( pTmpNode && GedNodeLevel( pTmpNode) >= uiStartLevel)
					{
						if( GedNodeLevel( pTmpNode) == uiStartLevel &&
							GedTagNum( pTmpNode) == FCS_ITERATOR_ATTRIBUTE)
						{
							if( RC_BAD( rc = GedGetUINT( pTmpNode, &uiAttrId)))
							{
								goto Exit;
							}

							puiPath[ uiPathPos++] = uiAttrId;
							if( uiPathPos > FCS_ITERATOR_MAX_PATH)
							{
								rc = RC_SET( FERR_SYNTAX);
								goto Exit;
							}
						}
						pTmpNode = pTmpNode->next;
					}
					puiPath[ uiPathPos] = 0;
				}

				/*
				Add the attribute path.
				*/

				if( RC_BAD( rc = FlmCursorAddFieldPath( hIterator,
					puiPath, 0)))
				{
					goto Exit;
				}

				break;
			}

			/*
			Add a numeric value to the criteria.
			*/

			case FCS_ITERATOR_NUMBER_VALUE:
			case FCS_ITERATOR_REC_PTR_VALUE:
			{
				/*
				To save conversion time, cheat to determine if
				the number is negative.
				*/

				FLMBYTE *	pucValue = (FLMBYTE *)GedValPtr( pCurNode);
				FLMBOOL		bNegative = ((*pucValue & 0xF0) == 0xB0)
									? (FLMBOOL)TRUE 
									: (FLMBOOL)FALSE;

				if( bNegative)
				{
					FLMINT32		i32Value;

					if( uiTag == FCS_ITERATOR_REC_PTR_VALUE)
					{
						rc = RC_SET( FERR_SYNTAX);
						goto Exit;
					}

					if( RC_BAD( rc = GedGetINT32( pCurNode, &i32Value)))
					{
						goto Exit;
					}

					if( RC_BAD( rc = FlmCursorAddValue( hIterator,
						FLM_INT32_VAL, &i32Value, 0)))
					{
						goto Exit;
					}
				}
				else
				{
					FLMUINT32	ui32Value;
					FLMUINT		uiValue;

					if( RC_BAD( rc = GedGetUINT32( pCurNode, &ui32Value)))
					{
						goto Exit;
					}

					if( uiTag == FCS_ITERATOR_NUMBER_VALUE)
					{
						if( RC_BAD( rc = FlmCursorAddValue( hIterator,
							FLM_UINT32_VAL, &ui32Value, 0)))
						{
							goto Exit;
						}
					}
					else if( uiTag == FCS_ITERATOR_REC_PTR_VALUE)
					{
						uiValue = ui32Value;
						if( RC_BAD( rc = FlmCursorAddValue( hIterator,
							FLM_REC_PTR_VAL, &uiValue, 0)))
						{
							goto Exit;
						}
					}
					else
					{
						/*
						Sanity check.
						*/

						flmAssert( 0);
					}
				}
				break;
			}

			/*
			Add a binary value to the criteria.
			*/

			case FCS_ITERATOR_BINARY_VALUE:
			{
				FLMBYTE *	pucValue = (FLMBYTE *)GedValPtr( pCurNode);
				FLMUINT		uiValLen = GedValLen( pCurNode);

				if( GedValType( pCurNode) != FLM_BINARY_TYPE)
				{
					rc = RC_SET( FERR_SYNTAX);
					goto Exit;
				}

				if( RC_BAD( rc = FlmCursorAddValue( hIterator,
					FLM_BINARY_VAL, pucValue, uiValLen)))
				{
					goto Exit;
				}
				break;
			}

			/*
			Add a UNICODE string value to the criteria.
			*/

			case FCS_ITERATOR_UNICODE_VALUE:
			{
				FLMUINT			uiLen;
				FLMUNICODE *	puzBuf;
				
				/*
				Mark the pool.
				*/

				pPoolMark = GedPoolMark( pPool);

				/*
				Determine the length of the string.
				*/

				if( RC_BAD( rc = GedGetUNICODE( pCurNode, NULL, &uiLen)))
				{
					goto Exit;
				}

				/*
				Allocate a temporary buffer.
				*/

				uiLen += 2;
				if( (puzBuf = (FLMUNICODE *)GedPoolAlloc( pPool, uiLen)) == NULL)
				{
					rc = RC_SET( FERR_MEM);
					goto Exit;
				}

				/*
				Extract the string and add it to the criteria.
				*/

				if( RC_BAD( rc = GedGetUNICODE( pCurNode, puzBuf, &uiLen)))
				{
					goto Exit;
				}

				if( RC_BAD( rc = FlmCursorAddValue( hIterator,
					FLM_UNICODE_VAL, puzBuf, 0)))
				{
					goto Exit;
				}

				GedPoolReset( pPool, pPoolMark);
				break;
			}

			/*
			Add a NATIVE, WP60, or Word String value to the criteria.
			*/

			case FCS_ITERATOR_NATIVE_VALUE:
			case FCS_ITERATOR_WP60_VALUE:
			case FCS_ITERATOR_WDSTR_VALUE:
			{
				FLMUINT		uiLen;
				FLMBYTE *	pucBuf;

				/*
				Mark the pool.
				*/

				pPoolMark = GedPoolMark( pPool);

				/*
				Determine the length of the string.
				*/

				if( uiTag == FCS_ITERATOR_NATIVE_VALUE)
				{
					if( RC_BAD( rc = GedGetNATIVE( pCurNode, NULL, &uiLen)))
					{
						goto Exit;
					}
				}
				else
				{
					rc = RC_SET( FERR_NOT_IMPLEMENTED);
					goto Exit;
				}

				/*
				Allocate a temporary buffer.
				*/

				uiLen += 2;
				if( (pucBuf = (FLMBYTE *)GedPoolAlloc( pPool, uiLen)) == NULL)
				{
					rc = RC_SET( FERR_MEM);
					goto Exit;
				}

				/*
				Extract the string and add it to the criteria.
				*/

				if( uiTag == FCS_ITERATOR_NATIVE_VALUE)
				{
					if( RC_BAD( rc = GedGetNATIVE( pCurNode, (char *)pucBuf, &uiLen)))
					{
						goto Exit;
					}

					if( RC_BAD( rc = FlmCursorAddValue( hIterator,
						FLM_STRING_VAL, pucBuf, 0)))
					{
						goto Exit;
					}
				}

				GedPoolReset( pPool, pPoolMark);
				break;
			}

			/*
			Add a native (internal) text value
			*/

			case FCS_ITERATOR_FLM_TEXT_VALUE:
			{
				if( RC_BAD( rc = FlmCursorAddValue( hIterator,
					FLM_TEXT_VAL, GedValPtr( pCurNode),
					GedValLen( pCurNode))))
				{
					goto Exit;
				}
				break;
			}

			/*
			Add an operator to the criteria.
			*/

			case FCS_ITERATOR_OPERATOR:
			{
				FLMUINT		uiOp;
				QTYPES		eTranslatedOp;

				/*
				Get the C/S operator ID.
				*/
				
				if( RC_BAD( rc = GedGetUINT( pCurNode, &uiOp)))
				{
					goto Exit;
				}

				if( !uiOp || 
					((uiOp - FCS_ITERATOR_OP_START) >= FCS_ITERATOR_OP_END))
				{
					rc = RC_SET( FERR_SYNTAX);
					goto Exit;
				}

				/*
				Translate the C/S ID to a FLAIM operator ID.
				*/
				
				if( RC_BAD( rc = fcsTranslateQCSToQFlmOp(
					uiOp, &eTranslatedOp)))
				{
					goto Exit;
				}

				/*
				Add the operator to the criteria.
				*/

				if( RC_BAD( rc = FlmCursorAddOp( hIterator, eTranslatedOp)))
				{
					goto Exit;
				}

				break;
			}

			default:
			{
				rc = RC_SET( FERR_SYNTAX);
				goto Exit;
			}
		}
		pCurNode = GedSibNext( pCurNode);
	}

Exit:

	return( rc);
}


/****************************************************************************
Desc:    Adds source information to an iterator.
*****************************************************************************/
FSTATIC RCODE fsvIteratorFromParse(
	FSV_WIRE *	pWire,
	POOL *		pPool)
{
	HFDB			hDb = HFDB_NULL;
	HFCURSOR		hIterator = pWire->getIteratorHandle();
	FLMUINT		uiIteratorId = FCS_INVALID_ID;
	NODE *		pFrom = pWire->getIteratorFrom();
	NODE *		pCurNode;
	NODE *		pCSAttrNode;
	NODE *		pTmpNode;
	RCODE			rc = FERR_OK;

	F_UNREFERENCED_PARM( pPool);

	/*
	If no "from" clause, jump to exit.
	*/

	if( !pFrom)
	{
		goto Exit;
	}

	/*
	Process each component of the "from" clause.
	*/

	if (hIterator == HFCURSOR_NULL)
	{
		FSV_SESN *	pSession;
		FLMUINT		uiContainerId = FLM_DATA_CONTAINER;
		FLMUINT		uiPath [4];

		uiPath [0] = FCS_ITERATOR_FROM;
		uiPath [1] = FCS_ITERATOR_CANDIDATE_SET;
		uiPath [2] = FCS_ITERATOR_RECORD_SOURCE;
		uiPath [3] = 0;
		if ((pCSAttrNode = GedPathFind( GED_TREE, pFrom, uiPath, 1)) == NULL)
		{
			rc = RC_SET( FERR_FAILURE);
			goto Exit;
		}

		/*
		Get the database handle.
		*/
		
		if( (pSession = pWire->getSession()) == NULL)
		{
			rc = RC_SET( FERR_BAD_HDL);
			goto Exit;
		}
		hDb = pSession->GetDatabase();

		/*
		Get the container ID.  A default value of
		FLM_DATA_CONTAINER will be used if a container ID
		is not found.
		*/

		if( (pTmpNode = GedFind( GED_TREE, pCSAttrNode,
			FCS_ITERATOR_CONTAINER_ID, 1)) != NULL)
		{
			if( RC_BAD( rc = GedGetUINT( pTmpNode,
				&uiContainerId)))
			{
				goto Exit;
			}
		}

		/*
		Initialize the cursor when we get the source - only one
		source is allowed.
		*/

		if( RC_BAD( rc = pSession->InitializeIterator(
			&uiIteratorId, hDb, uiContainerId, &hIterator)))
		{
			goto Exit;
		}

		/*
		Set the iterator handle and ID so they will be available
		for the parser to use.
		*/

		pWire->setIteratorId( uiIteratorId);
		pWire->setIteratorHandle( hIterator);
	}

	pCurNode = GedChild( pFrom);
	while( pCurNode)
	{
		switch( GedTagNum( pCurNode))
		{

			case FCS_ITERATOR_CANDIDATE_SET:
			{
				/*
				Process record sources and indexes.
				*/

				pCSAttrNode = GedChild( pCurNode);
				while( pCSAttrNode)
				{
					switch( GedTagNum( pCSAttrNode))
					{
						/*
						Define a record source.
						*/

						case FCS_ITERATOR_RECORD_SOURCE:
						{
							// Handled above.
							break;
						}

						/*
						Specify a FLAIM index.
						*/

						case FCS_ITERATOR_FLAIM_INDEX:
						{
							FLMUINT		uiIndexId;

							/*
							Get the index ID.
							*/

							if( RC_BAD( rc = GedGetUINT( pCSAttrNode, &uiIndexId)))
							{
								goto Exit;
							}

							/*
							Add the index.
							*/

							if( RC_BAD( rc = FlmCursorConfig( hIterator,
								FCURSOR_SET_FLM_IX, (void *)uiIndexId,
								(void *)0)))
							{
								goto Exit;
							}

							break;
						}

						/*
						Set the record type.
						*/

						case FCS_ITERATOR_RECORD_TYPE:
						{
							FLMUINT		uiRecordType;

							/*
							Get the record type.
							*/

							if( RC_BAD( rc = GedGetUINT( pCSAttrNode, &uiRecordType)))
							{
								goto Exit;
							}

							/*
							Add the record type.
							*/

							if( RC_BAD( rc = FlmCursorConfig( hIterator,
								FCURSOR_SET_REC_TYPE, (void *)uiRecordType,
								(void *)0)))
							{
								goto Exit;
							}
							break;
						}

						case FCS_ITERATOR_OK_TO_RETURN_KEYS:
						{
							FLMUINT		uiOkToReturnKeys;

							if( RC_BAD( rc = GedGetUINT( pCSAttrNode, &uiOkToReturnKeys)))
							{
								goto Exit;
							}

							if( RC_BAD( rc = FlmCursorConfig( hIterator,
								FCURSOR_RETURN_KEYS_OK, 
								(void *)(uiOkToReturnKeys ? 
									(FLMBOOL)TRUE : (FLMBOOL)FALSE), NULL)))
							{
								goto Exit;
							}
							break;
						}
					}
					pCSAttrNode = GedSibNext( pCSAttrNode);
				}
				break;
			}

			case FCS_ITERATOR_MODE:
			{
				FLMUINT		uiFlags;

				/*
				Get the mode flags.
				*/

				if( RC_BAD( rc = GedGetUINT( pCurNode, &uiFlags)))
				{
					goto Exit;
				}

				if( RC_BAD( rc = FlmCursorSetMode( hIterator, uiFlags)))
				{
					goto Exit;
				}
				break;
			}
		}
		pCurNode = GedSibNext( pCurNode);
	}

Exit:

	return( rc);
}


/****************************************************************************
Desc:    Adds a view to an iterator
*****************************************************************************/
FSTATIC RCODE fsvIteratorSelectParse(
	FSV_WIRE *	pWire,
	POOL *		pPool)
{
	NODE *		pSelect = pWire->getIteratorSelect();
	NODE *		pCurNode;
	NODE *		pView = NULL;
	FLMBOOL		bNullViewNotRec = FALSE;
	RCODE			rc = FERR_OK;

	F_UNREFERENCED_PARM( pPool);

	/*
	If no "select" clause, jump to exit.
	*/

	if( !pSelect)
	{
		goto Exit;
	}

	pCurNode = GedChild( pSelect);
	while( pCurNode)
	{
		switch( GedTagNum( pCurNode))
		{
			case FCS_ITERATOR_VIEW_TREE:
			{
				pView = GedChild( pCurNode);
				break;
			}

			case FCS_ITERATOR_NULL_VIEW_NOT_REC:
			{
				bNullViewNotRec = TRUE;
				break;
			}
		}
		pCurNode = GedSibNext( pCurNode);
	}

	/*
	Set the view record, if any (not supported).
	*/

	if( GedChild( pCurNode))
	{
		rc = RC_SET( FERR_NOT_IMPLEMENTED);
		goto Exit;
	}

Exit:

	return( rc);
}


/****************************************************************************
Desc:    Reads blocks from the database
*****************************************************************************/
FSTATIC RCODE fsvDbGetBlocks(
	HFDB				hDb,
	FLMUINT			uiAddress,
	FLMUINT			uiMinTransId,
	FLMUINT *		puiCount,
	FLMUINT *		puiBlocksExamined,
	FLMUINT *		puiNextBlkAddr,
	FLMUINT			uiFlags,
	POOL *			pPool,
	FLMBYTE **		ppBlocks,
	FLMUINT *		puiBytes)
{
	FDB *			pDb = (FDB *)hDb;
	FLMBOOL		bDbInitialized = FALSE;
	FLMBOOL		bTransStarted = FALSE;
	FLMUINT		uiLoop;
	FLMUINT		uiCount = *puiCount;
	SCACHE *		pSCache = NULL;
	FLMUINT		uiBlockSize;
	FLMUINT		uiMaxFileSize;
	RCODE			rc = FERR_OK;

	*ppBlocks = NULL;
	*puiCount = 0;
	*puiBlocksExamined = 0;
	*puiBytes = 0;

	if( IsInCSMode( hDb))
	{
		fdbInitCS( pDb);
		bDbInitialized = TRUE;

		CS_CONTEXT_p		pCSContext = pDb->pCSContext;
		FCL_WIRE				Wire( pCSContext, pDb);

		if( !pCSContext->bConnectionGood)
		{
			rc = RC_SET( FERR_BAD_SERVER_CONNECTION);
			goto Transmission_Error;
		}

		if( RC_BAD( rc = Wire.sendOp(
			FCS_OPCLASS_DATABASE, FCS_OP_DATABASE_GET_BLOCK)))
		{
			goto Exit;
		}

		if (RC_BAD( rc = Wire.sendNumber( WIRE_VALUE_ADDRESS,
			uiAddress)))
		{
			goto Transmission_Error;
		}

		if (RC_BAD( rc = Wire.sendNumber( WIRE_VALUE_TRANSACTION_ID,
			uiMinTransId)))
		{
			goto Transmission_Error;
		}

		if (RC_BAD( rc = Wire.sendNumber( WIRE_VALUE_COUNT,
			uiCount)))
		{
			goto Transmission_Error;
		}

		if (RC_BAD( rc = Wire.sendNumber( WIRE_VALUE_FLAGS,
			uiFlags)))
		{
			goto Transmission_Error;
		}

		if( RC_BAD( rc = Wire.sendTerminate()))
		{
			goto Transmission_Error;
		}

		/* Read the response. */
	
		if (RC_BAD( rc = Wire.read()))
		{
			goto Transmission_Error;
		}

		if( RC_BAD( rc = Wire.getRCode()))
		{
			if( rc != FERR_IO_END_OF_FILE)
			{
				goto Exit;
			}
		}

		*puiBlocksExamined = (FLMUINT)Wire.getNumber2();
		*puiCount = (FLMUINT)Wire.getCount();
		*puiNextBlkAddr = Wire.getAddress();

		if( *puiCount)
		{
			*puiBytes = Wire.getBlockSize();
			if( (*ppBlocks = (FLMBYTE *)GedPoolAlloc( pPool, *puiBytes)) == NULL)
			{
				rc = RC_SET( FERR_MEM);
				goto Exit;
			}

			f_memcpy( *ppBlocks, Wire.getBlock(), *puiBytes);
		}
		goto Exit;

Transmission_Error:
		pCSContext->bConnectionGood = FALSE;
		goto Exit;
	}

	if( !uiCount)
	{
		uiCount = 1;
	}

	uiBlockSize = pDb->pFile->FileHdr.uiBlockSize;
	uiMaxFileSize = pDb->pFile->uiMaxFileSize;
	bDbInitialized = TRUE;
	if ( RC_BAD( rc = fdbInit( pDb, FLM_READ_TRANS,
				FDB_TRANS_GOING_OK, 0, &bTransStarted)))
	{
		goto Exit;
	}

	if( (*ppBlocks = (FLMBYTE *)GedPoolAlloc( pPool, 
		uiBlockSize * uiCount)) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	/*
	Read uiCount blocks from the database starting at
	uiAddress.  If none of the blocks meet the min trans
	ID criteria, we will not return any blocks to the reader.
	*/

	*puiNextBlkAddr = BT_END;
	for( uiLoop = 0; uiLoop < uiCount; uiLoop++)
	{
		if( !FSAddrIsBelow( FSBlkAddress( 
			FSGetFileNumber( uiAddress), FSGetFileOffset( uiAddress)),
			pDb->LogHdr.uiLogicalEOF))
		{
			rc = RC_SET( FERR_IO_END_OF_FILE);
			goto Exit;
		}

		if( RC_BAD( rc = ScaGetBlock( pDb, NULL, BHT_FREE,
			uiAddress, NULL, &pSCache)))
		{
			goto Exit;
		}

		if( FB2UD( &pSCache->pucBlk[ BH_TRANS_ID]) >= uiMinTransId)
		{
			f_memcpy( (*ppBlocks + ((*puiCount) * uiBlockSize)),
				pSCache->pucBlk, uiBlockSize);

			(*puiCount)++;
			(*puiBytes) += uiBlockSize;
		}
		(*puiBlocksExamined)++;

		ScaReleaseCache( pSCache, FALSE);
		pSCache = NULL;

		uiAddress += uiBlockSize;
		if( FSGetFileOffset( uiAddress) >= uiMaxFileSize)
		{
			uiAddress = FSBlkAddress( FSGetFileNumber( uiAddress) + 1, 0);
		}
		*puiNextBlkAddr = uiAddress;
	}
	
Exit:

	if( pSCache)
	{
		ScaReleaseCache( pSCache, FALSE);
	}

	if( bTransStarted)
	{
		RCODE rc2 = flmAbortDbTrans( pDb);
		if ( RC_OK( rc))
		{
			rc = rc2;
		}
	}

	if( bDbInitialized)
	{
		fdbExit( pDb);
	}

	return( rc);
}

/****************************************************************************
Desc:    Commits a database transaction and updates the log header
*****************************************************************************/
RCODE  fsvDbTransCommitEx(
	HFDB				hDb,
	FSV_WIRE *		pWire)
{
	RCODE			rc = FERR_OK;
	FDB *			pDb = (FDB *)hDb;
	FLMBOOL		bIgnore;
	FLMBOOL		bForceCheckpoint = FALSE;
	FLMBYTE *	pucHeader = NULL;

	if( pWire->getFlags() & FCS_TRANS_FORCE_CHECKPOINT)
	{
		bForceCheckpoint = TRUE;
	}

	pucHeader = pWire->getBlock();

	if( IsInCSMode( hDb))
	{
		fdbInitCS( pDb);
		FCL_WIRE Wire( pDb->pCSContext, pDb);

		if (!pDb->pCSContext->bConnectionGood)
		{
			rc = RC_SET( FERR_BAD_SERVER_CONNECTION);
		}
		else
		{
			rc = Wire.doTransOp(
				FCS_OP_TRANSACTION_COMMIT_EX, 0, 0, 0,
				pucHeader, bForceCheckpoint);
		}
		goto Exit;
	}

	if (RC_BAD( rc = fdbInit( pDb, FLM_NO_TRANS,
										FDB_TRANS_GOING_OK, 0, &bIgnore)))
	{
		goto Exit;
	}

	/*
	If there is an invisible transaction going, it should not be
	commitable by an application.
	*/

	if ((pDb->uiTransType == FLM_NO_TRANS) ||
		 (pDb->uiFlags & FDB_INVISIBLE_TRANS))
	{
		rc = RC_SET( FERR_NO_TRANS_ACTIVE);
		goto Exit;
	}

	/*
	See if we have a transaction going which should be aborted.
	*/

	if( RC_BAD( pDb->AbortRc))
	{
		rc = RC_SET( FERR_ABORT_TRANS);
		goto Exit;
	}

	/*
	Fix up the log header.  Currently, only fields directly
	related to a backup operation are updated.
	*/

	if( pucHeader)
	{
		FLMBYTE *	pLogHdr = &pucHeader[ 16];
		FLMBYTE *	pucUncommittedHdr = &pDb->pFile->ucUncommittedLogHdr [0];

		f_memcpy( &pucUncommittedHdr [LOG_LAST_BACKUP_TRANS_ID],
			&pLogHdr[ LOG_LAST_BACKUP_TRANS_ID], 4);

		f_memcpy( &pucUncommittedHdr [LOG_BLK_CHG_SINCE_BACKUP],
			&pLogHdr[ LOG_BLK_CHG_SINCE_BACKUP], 4);

		f_memcpy( &pucUncommittedHdr [LOG_INC_BACKUP_SEQ_NUM],
			&pLogHdr[ LOG_INC_BACKUP_SEQ_NUM], 4);

		f_memcpy( &pucUncommittedHdr [LOG_INC_BACKUP_SERIAL_NUM],
			&pLogHdr[ LOG_INC_BACKUP_SERIAL_NUM], F_SERIAL_NUM_SIZE);
	}

	/*
	Commit the transaction
	*/

	rc = flmCommitDbTrans( pDb, 0, bForceCheckpoint);

Exit:

	flmExit( FLM_DB_TRANS_COMMIT, pDb, rc);
	return( rc);
}

/****************************************************************************
Desc:    Looks up session, database, and iterator handles.
*****************************************************************************/
FSTATIC RCODE fsvGetHandles(
	FSV_WIRE *	pWire)
{
	FSV_SCTX *	pServerContext = NULL;
	FSV_SESN *	pSession = NULL;
	HFCURSOR		hIterator = HFCURSOR_NULL;
	RCODE			rc = FERR_OK;

	if( RC_BAD( rc = fsvGetGlobalContext( &pServerContext)))
	{
		goto Exit;
	}

	if( pWire->getSessionId() != FCS_INVALID_ID)
	{
		if( RC_BAD( pServerContext->GetSession( pWire->getSessionId(),
			&pSession)))
		{
			rc = RC_SET( FERR_BAD_HDL);
			goto Exit;
		}

		if( pSession->getCookie() != pWire->getSessionCookie())
		{
			rc = RC_SET( FERR_BAD_HDL);
			goto Exit;
		}

		pWire->setSession( pSession);
	}

	if( pSession)
	{
		pWire->setFDB( (FDB *)pSession->GetDatabase());
		if( pWire->getIteratorId() != FCS_INVALID_ID)
		{
			if( RC_BAD( rc = pSession->GetIterator(
				pWire->getIteratorId(), &hIterator)))
			{
				goto Exit;
			}

			pWire->setIteratorHandle( hIterator);
		}
	}

Exit:

#ifdef FSV_LOGGING
	if( RC_BAD( rc))
	{
		fsvLogHandlerMessage( NULL,
			(FLMBYTE *)"Error finding requested handles.", rc, FSV_LOG_DEBUG);
	}
#endif

	return( rc);
}


#ifdef FSV_LOGGING
/******************************************************************************
Desc: Logs a message
*****************************************************************************/
void fsvLogHandlerMessage(
	FSV_SESN *	pSession,
	FLMBYTE *	pucMsg,
	RCODE			rc,
	FLMUINT		uiMsgSeverity)
{
	FSV_SCTX *  pServerContext = NULL;

	if( pucMsg && pucMsg[ 0])
	{
		if( RC_BAD( fsvGetGlobalContext( &pServerContext)))
		{
			goto Exit;
		}

		pServerContext->LogMessage( pSession, pucMsg, rc, uiMsgSeverity);
	}

Exit:

	return;
}
#endif   // FSV_LOGGING

/****************************************************************************
Desc:    
*****************************************************************************/
RCODE fsvPostStreamedRequest(
	FSV_SESN *	pSession,
	FLMBYTE *	pucPacket,
	FLMUINT		uiPacketSize,
	FLMBOOL		bLastPacket,
	FCS_BIOS *	pSessionResponse)
{
	FLMBOOL		bReleaseSession = FALSE;
	RCODE			rc = FERR_OK;
	POOL			localPool;

	GedPoolInit( &localPool, 1024);

	if( !pSession && !bLastPacket)
	{
		/*
		If this is a session open request, the request must
		be contained in a single packet.
		*/

		rc = RC_SET( FERR_ILLEGAL_OP);
		goto Exit;
	}

	if( !pSession)
	{
		FCS_BIOS		biosInput;
		FCS_DIS		dataIStream;
		FCS_DOS		dataOStream;

		if( RC_BAD( rc = dataIStream.setup( &biosInput)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = dataOStream.setup( pSessionResponse)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = biosInput.write( pucPacket, uiPacketSize)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = fsvProcessRequest( &dataIStream, 
			&dataOStream, &localPool, NULL)))
		{
			goto Exit;
		}
	}
	else
	{
		FCS_BIOS	*	pServerBIStream;
		FCS_BIOS	*	pServerBOStream;

		/*
		Need to add a reference to the session object so that if the request closes
		the session, the response stream will not be destructed until the response
		has been returned to the client.
		*/

		pSession->AddRef();
		bReleaseSession = TRUE;

		if( RC_BAD( rc = pSession->GetBIStream( &pServerBIStream)))
		{
			goto Exit;
		}
		
		if( RC_BAD( rc = pSession->GetBOStream( &pServerBOStream)))
		{
			goto Exit;
		}
		
		if( RC_BAD( rc = pServerBIStream->write( pucPacket, uiPacketSize)))
		{
			goto Exit;
		}

		if( bLastPacket)
		{
			FCS_DIS		dataIStream;
			FCS_DOS		dataOStream;

			if( RC_BAD( rc = dataIStream.setup( pServerBIStream)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = dataOStream.setup( pServerBOStream)))
			{
				goto Exit;
			}

			GedPoolReset( pSession->getWireScratchPool(), NULL);
			if( RC_BAD( rc = fsvProcessRequest( &dataIStream, &dataOStream,
				pSession->getWireScratchPool(), NULL)))
			{
				goto Exit;
			}
		}
	}

Exit:

	GedPoolFree( &localPool);

	if( bReleaseSession)
	{
		pSession->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:    
*****************************************************************************/
RCODE fsvGetStreamedResponse(
	FSV_SESN *	pSession,
	FLMBYTE *	pucPacketBuffer,
	FLMUINT		uiMaxPacketSize,
	FLMUINT *	puiPacketSize,
	FLMBOOL *	pbLastPacket)
{
	FCS_BIOS *		pServerBOStream = NULL;
	RCODE				rc = FERR_OK;
	
	if( RC_BAD( rc = pSession->GetBOStream( &pServerBOStream)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pServerBOStream->read( pucPacketBuffer,
		uiMaxPacketSize, puiPacketSize)))
	{
		if( rc == FERR_EOF_HIT)
		{
			*pbLastPacket = TRUE;
			rc = FERR_OK;
		}
		goto Exit;
	}

	if( !pServerBOStream->isDataAvailable())
	{
		*pbLastPacket = TRUE;
	}

Exit:

	return( rc);
}
