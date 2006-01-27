//-------------------------------------------------------------------------
// Desc:	Routines for updating statistics - for monitoring - definitions.
// Tabs:	3
//
//		Copyright (c) 1997-2006 Novell, Inc. All Rights Reserved.
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
// $Id: flmstat.h 12263 2006-01-19 14:43:23 -0700 (Thu, 19 Jan 2006) dsanders $
//-------------------------------------------------------------------------

/****************************************************************************
File:    flmstat.h
Title:   FLAIM Statistics Include File
Owner:   FLAIM Team
Tabs:    4,3

	Copyright (C) 1997-1999, Novell Inc., 
	All Rights Reserved, Patents Pending
	COMPANY CONFIDENTIAL -- NOT FOR DISTRIBUTION
-----------------------------------------------------------------------------
Desc: This include file contains the structure definitions and prototypes
		needed to capture statistics.

 * $Log$
 * Revision 4.19  2005/11/10 22:53:22  dsanders
 * Check in open-source changes at head branch - taken from rosalind base.
 *
 * Revision 4.18.2.1  2005/11/09 22:49:30  dsanders
 * Scrubbed for open sourcing.
 *
 * Revision 4.18  2004/06/15 20:36:47  ahodgkinson
 * Merged encryption code from Kepler-e.  Also Fixed 64-bit Linux build and
 * NetWare builds (cw and wc).
 *
 * Revision 4.17.2.1  2004/04/14 18:46:14  dsanders
 * Added include of fpackon.h and fpackoff.h
 *
 * Revision 4.17  2003/12/02 23:28:14  ahodgkinson
 * Changes to support a user-provided memory allocator.  Also restructured
 * the code so that flaim.h is the only header file that consumers of FLAIM
 * need in order to use FLAIM.
 *
 * Revision 4.16  2003/06/04 21:25:58  ahodgkinson
 * Fixed RCS log messages.
 *
 *    Rev 4.15   10 Dec 2002 08:35:36   dss
 * Defect 322292.  Adding of more statistics.
 * 
 *    Rev 4.14   21 Nov 2002 09:46:42   dss
 * Defect 320562.  Changes to allow 64 bits for statistics data - monitoring.
 * 
 *    Rev 4.13   08 Mar 2002 17:11:42   andy
 * Misc. cleanup.
 * 
 *    Rev 4.12   05 Mar 2002 10:53:42   dss
 * Modified so that DB_STATS holds a pointer to the full database name instead
 * of just having the base name of the database.
 * 
 *    Rev 4.11   15 Feb 2002 15:34:06   dss
 * Got rid of unneeded statistic.
 * 
 *    Rev 4.10   15 Feb 2002 09:43:34   dss
 * Got rid of statistics that are no longer relevant.
 * 
 *    Rev 4.9   11 Feb 2002 11:11:42   dss
 * Changed how we save queries when they are finished.  No longer stored in
 * the statistics structure.
 * 
 *    Rev 4.8   16 Jan 2002 17:06:12   dss
 * Added includes of fpackon.h and fpackoff.h
 * 
 *    Rev 4.7   07 Jan 2002 10:59:44   andy
 * Defect 287732.  Support for persistent index suspend.
 * 
 *    Rev 4.6   03 May 2001 14:59:00   dss
 * Changed to use LFH in names and macros instead of LFSA2_0 or LSA.
 * 
 *    Rev 4.5   28 Jun 2000 13:49:14   dss
 * Got rid of specificity scores.
 * 
 *    Rev 4.4   26 Jun 2000 15:36:30   blj
 * inline optimizations.
 * 
 *    Rev 4.3   19 Jun 2000 11:56:26   andy
 * Renamed flmIntShareGetStats to flmIntGetStats.
 * 
 *    Rev 4.2   16 Jun 2000 17:41:08   blj
 * removed share structure.
 * 
 *    Rev 4.1   16 Jun 2000 16:54:12   blj
 * Version 4.1
 * 
 *    Rev 3.19   02 Jun 2000 14:41:24   swp
 * Removed cplusplus
 * 
 *    Rev 3.18   31 May 2000 13:43:14   dss
 * Changed ui32StartTime to uiStartTime and ui32StopTime to uiStopTime.
 * 
 *    Rev 3.17   26 May 2000 16:48:32   swp
 * Removed wpd.h and sem includes.
 * 
 *    Rev 3.16   14 Apr 2000 15:54:00   dss
 * Defect #232010.  Got rid of some FSTAT_xxx tags that are no longer used.
 * 
 *    Rev 3.15   10 Feb 2000 08:28:00   blj
 * Defect #223496 - added wpd.h
 * 
 *    Rev 3.14   21 Dec 1999 15:53:22   dss
 * Defect #212233.  Took cache hit, cache fault stuff out of block io stats.
 * Also added tags for new memory information stuff.
 * 
 *    Rev 3.13   08 Nov 1999 17:03:18   blj
 * Fix link error.
 * 
 *    Rev 3.12   04 Nov 1999 14:38:08   dss
 * Got rid of uiNumOPCs using variable.
 * 
 *    Rev 3.11   04 Nov 1999 09:41:40   dss
 * Got rid of some statistics that we aren't going to keep anymore.
 * 
 *    Rev 3.10   08 Oct 1999 09:06:22   andy
 * Added new tags for record cache statistics.
 * 
 *    Rev 3.9   09 Sep 1999 13:26:00   andy
 * Renamed STORE_STATS to DB_STATS.
 * 
 *    Rev 3.8   12 May 1999 09:50:34   dss
 * SPD #236021.  Took uiStoreNum parameter out of flmStatGetStore function.
 * Also renamed uiStoreAllocSeq to be uiDBAllocSeq.
 * 
 *    Rev 3.7   11 May 1999 13:06:22   dss
 * SPD #236021.  Changed types to be more optimal.
 * 
 *    Rev 3.6   15 Jan 1999 16:18:04   dss
 * Bugs 222355 & 217694.  Added code for bDidAuxTree.
 * 
 *    Rev 3.5   21 Dec 1998 17:23:30   dss
 * Made flmQueryStatsToGed a public routine - added prototype.
 * 
 *    Rev 3.4   21 Dec 1998 13:57:50   swp
 * do match changes.
 * 
 *    Rev 3.3   30 Nov 1998 16:40:50   dss
 * Added stuff for collecting query statistics.
 * 
 *    Rev 3.2   30 Nov 1998 13:17:00   blj
 * Updated copyright notice.
 * 
 *    Rev 3.1   15 Sep 1998 16:39:04   andy
 * Changes to align structures.
 * 
 *    Rev 3.0   26 Feb 1998 13:43:26   dss
 * New file for SLICK project
 * 
 *    Rev 2.0   12 Mar 1997 14:05:14   dss
 * GroupWise 5.1 Shipping Code
 * 
 *    Rev 1.0   10 Jan 1997 08:29:58   dss
 * Initial revision.

*****************************************************************************/
#ifndef FLMSTAT_H
#define FLMSTAT_H

#include "fpackon.h"
// IMPORTANT NOTE: No other include files should follow this one except
// for fpackoff.h

/**************************************************************************
				Various function prototypes.
**************************************************************************/

RCODE	flmStatGetDb(
	FLM_STATS *			pFlmStats,
	void *				pFile,
	FLMUINT				uiLowStart,
	DB_STATS **			ppDbStatsRV,
	FLMUINT *			puiDbAllocSeqRV,
	FLMUINT *			puiDbTblPosRV);

RCODE	flmStatGetLFile(
	DB_STATS *			pDbStats,
	FLMUINT				uiLFileNum,
	FLMUINT				uiLfType,
	FLMUINT				uiLowStart,
	LFILE_STATS **		ppLFileStatsRV,
	FLMUINT *			puiLFileAllocSeqRV,
	FLMUINT *			puiLFileTblPosRV);

void flmStatReset(
	FLM_STATS *			pStats,
	FLMBOOL				bSemAlreadyLocked,
	FLMBOOL				bFree);

void flmStatStart(
	FLM_STATS *			pStats);

void flmStatStop(
	FLM_STATS *			pStats);

RCODE flmStatInit(
	FLM_STATS *			pStats,
	FLMBOOL				bEnableSharing);

void flmUpdateBlockIOStats(
	BLOCKIO_STATS *	pDest,
	BLOCKIO_STATS *	pSrc);

RCODE	flmStatUpdate(
	FLM_STATS *			pDestStats,
	FLM_STATS *			pSrcStats);

void flmFreeSavedQueries(
	FLMBOOL				bMutexAlreadyLocked);

void flmSaveQuery(
	HFCURSOR				hCursor);

BLOCKIO_STATS * flmGetBlockIOStatPtr(
	DB_STATS *			pDbStats,
	LFILE_STATS *		pLFileStats,
	FLMBYTE *			pBlk,
	FLMUINT				uiBlkType);

void flmAddElapTime(
	F_TMSTAMP  *		pStartTime,
	FLMUINT64 *			pui64ElapMilli);

/****************************************************************************
Desc:	This routine updates statistics from one DISKIO_STAT structure
		into another.
****************************************************************************/
FINLINE void flmUpdateDiskIOStats(
	DISKIO_STAT *		pDest,
	DISKIO_STAT *		pSrc)
{
	pDest->ui64Count += pSrc->ui64Count;
	pDest->ui64TotalBytes += pSrc->ui64TotalBytes;
	pDest->ui64ElapMilli += pSrc->ui64ElapMilli;
}

/****************************************************************************
Desc:
****************************************************************************/
FINLINE void flmUpdateCountTimeStats(
	COUNT_TIME_STAT *	pDest,
	COUNT_TIME_STAT *	pSrc)
{
	pDest->ui64Count += pSrc->ui64Count;
	pDest->ui64ElapMilli += pSrc->ui64ElapMilli;
}

#include "fpackoff.h"

#endif
