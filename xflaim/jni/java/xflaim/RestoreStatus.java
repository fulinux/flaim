//------------------------------------------------------------------------------
// Desc:	Restore Status
//
// Tabs:	3
//
//		Copyright (c) 2004-2006 Novell, Inc. All Rights Reserved.
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
// $Id: RestoreStatus.java 3110 2006-01-19 13:09:08 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------
package xflaim;

/**
 * This interface allows XFlaim's backup subsystem to periodicly pass
 * information about the status of a restore operation (bytes completed and
 * bytes remaining) while the operation is running.  The implementor may do
 * anything it wants with the information, such as using it to update a
 * progress bar or simply ignoring it.
 */
public interface RestoreStatus
{
	/**
	 * 
	 * @param eAction
	 * @param lBytesToDo
	 * @param lBytesDone
	 * @return Returns a status code.  The integer should one of the constants
	 * found in {@link xflaim.RCODE xflaim.RCODE}.
	 * Note that returning anything other than NE_XFLM_OK will cause the
	 * restore operation to abort and an XFLaimException to be thrown.
	 */
	int reportProgress(
		int		eAction,
		long		lBytesToDo,
		long		lBytesDone);

	/**
	 * 
	 * @param eAction
	 * @param eErrCode
	 * @return Returns a status code.  The integer should one of the constants
	 * found in {@link xflaim.RCODE xflaim.RCODE}.
	 * Note that returning anything other than NE_XFLM_OK will cause the
	 * restore operation to abort and an XFLaimException to be thrown.
	 */
	int reportError(
		int		eAction,
		int		eErrCode);

	/**
	 * 
	 * @param eAction
	 * @param lTransId
	 * @param iStartTime
	 * @return Returns a status code.  The integer should one of the constants
	 * found in {@link xflaim.RCODE xflaim.RCODE}.
	 * Note that returning anything other than NE_XFLM_OK will cause the
	 * restore operation to abort and an XFLaimException to be thrown.
	 */
	int reportBeginTrans(
		int		eAction,
		long		lTransId,
		int		iStartTime);

	/**
	 * 
	 * @param eAction
	 * @param lTransId
	 * @return Returns a status code.  The integer should one of the constants
	 * found in {@link xflaim.RCODE xflaim.RCODE}.
	 * Note that returning anything other than NE_XFLM_OK will cause the
	 * restore operation to abort and an XFLaimException to be thrown.
	 */
	int reportCommitTrans(
		int		eAction,
		long		lTransId);

	/**
	 * 
	 * @param eAction
	 * @param lTransId
	 * @return Returns a status code.  The integer should one of the constants
	 * found in {@link xflaim.RCODE xflaim.RCODE}.
	 * Note that returning anything other than NE_XFLM_OK will cause the
	 * restore operation to abort and an XFLaimException to be thrown.
	 */
	int reportAbortTrans(
		int		eAction,
		long		lTransId);

	/**
	 * 
	 * @param eAction
	 * @param lTransId
	 * @param iLfNum
	 * @param Key
	 * @return Returns a status code.  The integer should one of the constants
	 * found in {@link xflaim.RCODE xflaim.RCODE}.
	 * Note that returning anything other than NE_XFLM_OK will cause the
	 * restore operation to abort and an XFLaimException to be thrown.
	 */
	int reportRemoveData(
		int		eAction,
		long		lTransId,
		int		iLfNum,
		byte[]	Key);

	/**
	 * 
	 * @param eAction
	 * @param lTransId
	 * @param iLfNum
	 * @param Key
	 * @return Returns a status code.  The integer should one of the constants
	 * found in {@link xflaim.RCODE xflaim.RCODE}.
	 * Note that returning anything other than NE_XFLM_OK will cause the
	 * restore operation to abort and an XFLaimException to be thrown.
	 */
	int reportInsertData(
		int		eAction,
		long		lTransId,
		int		iLfNum,
		byte[]	Key);

	/**
	 * 
	 * @param eAction
	 * @param lTransId
	 * @param iLfNum
	 * @param pucKey
	 * @return Returns a status code.  The integer should one of the constants
	 * found in {@link xflaim.RCODE xflaim.RCODE}.
	 * Note that returning anything other than NE_XFLM_OK will cause the
	 * restore operation to abort and an XFLaimException to be thrown.
	 */
	int reportReplaceData(
		int		eAction,
		long		lTransId,
		int		iLfNum,
		byte		pucKey);

	/**
	 * 
	 * @param eAction
	 * @param lTransId
	 * @param iLfNum
	 * @return Returns a status code.  The integer should one of the constants
	 * found in {@link xflaim.RCODE xflaim.RCODE}.
	 * Note that returning anything other than NE_XFLM_OK will cause the
	 * restore operation to abort and an XFLaimException to be thrown.
	 */
	int reportLFileCreate(
		int		eAction,
		long		lTransId,
		int		iLfNum);

	/**
	 * 
	 * @param eAction
	 * @param lTransId
	 * @param iLfNum
	 * @param iRootBlk
	 * @param lNextNodeId
	 * @param lFirstDocId
	 * @param lLastDocId
	 * @return Returns a status code.  The integer should one of the constants
	 * found in {@link xflaim.RCODE xflaim.RCODE}.
	 * Note that returning anything other than NE_XFLM_OK will cause the
	 * restore operation to abort and an XFLaimException to be thrown.
	 */
	int reportLFileUpdate(
		int		eAction,
		long		lTransId,
		int		iLfNum,
		int		iRootBlk,
		long		lNextNodeId,
		long		lFirstDocId,
		long		lLastDocId);

	/**
	 * 
	 * @param eAction
	 * @param lTransId
	 * @param iDictType
	 * @param iDictNum
	 * @param bDeleting
	 * @return Returns a status code.  The integer should one of the constants
	 * found in {@link xflaim.RCODE xflaim.RCODE}.
	 * Note that returning anything other than NE_XFLM_OK will cause the
	 * restore operation to abort and an XFLaimException to be thrown.
	 */
	int reportUpdateDict(
		int		eAction,
		long		lTransId,
		int		iDictType,
		int		iDictNum,
		boolean	bDeleting);
	
	/**
	 * 
	 * @param eAction
	 * @param lTransId
	 * @param iMaintDocNum
	 * @param iStartBlkAddr
	 * @param iEndBlkAddr
	 * @param iCount
	 * @return Returns a status code.  The integer should one of the constants
	 * found in {@link xflaim.RCODE xflaim.RCODE}.
	 * Note that returning anything other than NE_XFLM_OK will cause the
	 * restore operation to abort and an XFLaimException to be thrown.
	 */
	int reportBlockChainFree(
		int		eAction,
		long		lTransId,
		int		iMaintDocNum,
		int		iStartBlkAddr,
		int		iEndBlkAddr,
		int		iCount);

	/**
	 * 
	 * @param eAction
	 * @param lTransId
	 * @param iIndexNum
	 * @return Returns a status code.  The integer should one of the constants
	 * found in {@link xflaim.RCODE xflaim.RCODE}.
	 * Note that returning anything other than NE_XFLM_OK will cause the
	 * restore operation to abort and an XFLaimException to be thrown.
	 */
	int reportIndexSuspend(
		int		eAction,
		long		lTransId,
		int		iIndexNum);

	/**
	 * 
	 * @param eAction
	 * @param lTransId
	 * @param iIndexNum
	 * @return Returns a status code.  The integer should one of the constants
	 * found in {@link xflaim.RCODE xflaim.RCODE}.
	 * Note that returning anything other than NE_XFLM_OK will cause the
	 * restore operation to abort and an XFLaimException to be thrown.
	 */
	int reportIndexResume(
		int		eAction,
		long		lTransId,
		int		iIndexNum);

	/**
	 * 
	 * @param eAction
	 * @param lTransId
	 * @param iCount
	 * @return Returns a status code.  The integer should one of the constants
	 * found in {@link xflaim.RCODE xflaim.RCODE}.
	 * Note that returning anything other than NE_XFLM_OK will cause the
	 * restore operation to abort and an XFLaimException to be thrown.
	 */
	int reportReduce(
		int		eAction,
		long		lTransId,
		int		iCount);

	/**
	 * 
	 * @param eAction
	 * @param lTransId
	 * @param iOldDbVersion
	 * @param iNewDbVersion
	 * @return Returns a status code.  The integer should one of the constants
	 * found in {@link xflaim.RCODE xflaim.RCODE}.
	 * Note that returning anything other than NE_XFLM_OK will cause the
	 * restore operation to abort and an XFLaimException to be thrown.
	 */
	int reportUpgrade(
		int		eAction,
		long		lTransId,
		int		iOldDbVersion,
		int		iNewDbVersion);

	/**
	 * 
	 * @param eAction
	 * @param iFileNum
	 * @return Returns a status code.  The integer should one of the constants
	 * found in {@link xflaim.RCODE xflaim.RCODE}.
	 * Note that returning anything other than NE_XFLM_OK will cause the
	 * restore operation to abort and an XFLaimException to be thrown.
	 */
	int reportOpenRflFile(
		int		eAction,
		int		iFileNum);

	/**
	 * 
	 * @param eAction
	 * @param iFileNum
	 * @param iBytesRead
	 * @return Returns a status code.  The integer should one of the constants
	 * found in {@link xflaim.RCODE xflaim.RCODE}.
	 * Note that returning anything other than NE_XFLM_OK will cause the
	 * restore operation to abort and an XFLaimException to be thrown.
	 */
	int reportRflRead(
		int		eAction,
		int		iFileNum,
		int		iBytesRead);
}
