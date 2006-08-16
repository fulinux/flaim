//------------------------------------------------------------------------------
// Desc:	Query object
//
// Tabs:	3
//
//		Copyright (c) 2003-2006 Novell, Inc. All Rights Reserved.
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
// $Id$
//------------------------------------------------------------------------------

package xflaim;

/**
 * This class encapsulates the XFlaim IF_Query interface.
 */
public class Query 
{
	
	Query(
		Db 			jdb,
		int			iCollection) throws XFlaimException
	{

		m_this = _createQuery( iCollection);
		
		if (jdb == null)
		{
			throw new XFlaimException( -1, "No legal jDb reference");
		}
		
		m_jdb = jdb;
	}
	
	protected void finalize()
	{
		// The F_Query and F_Db classes are not thread-safe.  The proper way
		// of using XFlaim is to get a new instance of Db for each thread.
		// Unfortunately, the garbage collector runs in its own thread.  This
		// leads to a potential race condition down in the C++ code when one
		// thread tries to create an already existing query (which results in a
		// call to F_Query::AddRef()) and the GC tries to destroy the same
		// query (which results in a call to F_Query::Release()).
		// We protect against this by synchronizing against the instance of
		// Db.  Note that we are not protecting any of the accesses to the
		// query; only creating and destroying.  Query and Db are still
		// not completely thread-safe.
		
		synchronized( m_jdb)
		{
			// Release the associated DOMNode.
			
			if (m_this != 0)
			{
				_release( m_this);
			}
		}
		
		// Free our reference to the Db object.
		
		m_jdb = null;
	}
	
	public long getThis()
	{
		return m_this;
	}
	
	public void setLanguage(
		int	iLanguage) throws XFlaimException
	{
		_setLanguage( m_this, iLanguage);
	}
	
	public void setupQueryExpr(
		String	sQuery) throws XFlaimException
	{
		_setupQueryExpr( m_this, m_jdb.m_this, sQuery);
	}
	
	public void copyCriteria(
		Query	queryToCopy) throws XFlaimException
	{
		_copyCriteria( m_this, queryToCopy.m_this);
	}
	
	public void addXPathComponent(
		long		lThis,
		int		iXPathAxis,
		int		iNodeType,
		int		iNameId) throws XFlaimException
	{
		_addXPathComponent( m_this, iXPathAxis, iNodeType, iNameId);
	}
	
	public void addOperator(
		int				iOperator) throws XFlaimException
	{
		_addOperator( m_this, iOperator);
	}
		
	public void addStringOperator(
		int				iOperator,
		boolean			bCaseInsensitive,
		boolean			bCompressWhitespace,
		boolean			bNoWhitespace,
		boolean			bNoUnderscores,
		boolean			bNoDashes,
		boolean			bWhitespaceAsSpace,
		boolean			bIgnoreLeadingSpace,
		boolean			bIgnoreTrailingSpace) throws XFlaimException
	{
		_addStringOperator( m_this, iOperator, bCaseInsensitive, bCompressWhitespace,
				bNoWhitespace, bNoUnderscores, bNoDashes, bWhitespaceAsSpace,
				bIgnoreLeadingSpace, bIgnoreTrailingSpace);
	}
	
	public void addStringValue(
		String	sValue) throws XFlaimException
	{
		_addStringValue( m_this, sValue);
	}
	
	public void addBinaryValue(
		byte []	Value) throws XFlaimException
	{
		_addBinaryValue( m_this, Value);
	}
	
	public void addLongValue(
		long		lValue) throws XFlaimException
	{
		_addLongValue( m_this, lValue);
	}
	
	public void addBoolean(
		boolean	bValue,
		boolean	bUnknown) throws XFlaimException
	{
		_addBoolean( m_this, bValue, bUnknown);
	}

// PRIVATE METHODS

	private native void _release(
		long		lThis);
		
	private native long _createQuery(
		int		iCollection);

	private native void _setLanguage(
		long	lThis,
		int	iLanguage) throws XFlaimException;
		
	private native void _setupQueryExpr(
		long		lThis,
		long		lDbRef,
		String	sQuery) throws XFlaimException;
		
	private native void _copyCriteria(
		long		lThis,
		long		lQueryToCopy) throws XFlaimException;

	private native void _addXPathComponent(
		long		lThis,
		int		iXPathAxis,
		int		iNodeType,
		int		iNameId) throws XFlaimException;
	
	private native void _addOperator(
		long				lThis,
		int				iOperator) throws XFlaimException;
		
	private native void _addStringOperator(
		long				lThis,
		int				iOperator,
		boolean			bCaseInsensitive,
		boolean			bCompressWhitespace,
		boolean			bNoWhitespace,
		boolean			bNoUnderscores,
		boolean			bNoDashes,
		boolean			bWhitespaceAsSpace,
		boolean			bIgnoreLeadingSpace,
		boolean			bIgnoreTrailingSpace) throws XFlaimException;
	
	private native void _addStringValue(
		long		lThis,
		String	sValue) throws XFlaimException;
	
	private native void _addBinaryValue(
		long		lThis,
		byte []	Value) throws XFlaimException;
	
	private native void _addLongValue(
		long		lThis,
		long		lValue) throws XFlaimException;
	
	private native void _addBoolean(
		long		lThis,
		boolean	bValue,
		boolean	bUnknown) throws XFlaimException;

	private long	m_this;
	private Db		m_jdb;
}

/*

FUNCTIONS NOT YET IMPLEMENTED

virtual RCODE FLMAPI addFunction(
	eQueryFunctions		eFunction) = 0;

virtual RCODE FLMAPI addFunction(
	IF_QueryValFunc *		pFuncObj,
	FLMBOOL					bHasXPathExpr) = 0;

virtual RCODE FLMAPI getFirst(
	IF_Db *					pDb,
	IF_DOMNode **			ppNode,
	FLMUINT					uiTimeLimit = 0) = 0;	// milliseconds

virtual RCODE FLMAPI getLast(
	IF_Db *					pDb,
	IF_DOMNode **			ppNode,
	FLMUINT					uiTimeLimit = 0) = 0;	// milliseconds

virtual RCODE FLMAPI getNext(
	IF_Db *					pDb,
	IF_DOMNode **			ppNode,
	FLMUINT					uiTimeLimit = 0,		// milliseconds
	FLMUINT					uiNumToSkip = 0,
	FLMUINT *				puiNumSkipped = NULL) = 0;

virtual RCODE FLMAPI getPrev(
	IF_Db *					pDb,
	IF_DOMNode **			ppNode,
	FLMUINT					uiTimeLimit = 0,		// milliseconds
	FLMUINT					uiNumToSkip = 0,
	FLMUINT *				puiNumSkipped = NULL) = 0;

virtual RCODE FLMAPI getCurrent(
	IF_Db *					pDb,
	IF_DOMNode **			ppNode) = 0;

virtual void FLMAPI resetQuery( void) = 0;

virtual RCODE FLMAPI getStatsAndOptInfo(
	FLMUINT *				puiNumOptInfos,
	XFLM_OPT_INFO **		ppOptInfo) = 0;

virtual void FLMAPI freeStatsAndOptInfo(
	XFLM_OPT_INFO **		ppOptInfo) = 0;

virtual void FLMAPI setDupHandling(
	FLMBOOL					bRemoveDups) = 0;

virtual RCODE FLMAPI setIndex(
	FLMUINT					uiIndex) = 0;

virtual RCODE FLMAPI getIndex(
	IF_Db *					pDb,
	FLMUINT *				puiIndex,
	FLMBOOL *				pbHaveMultiple) = 0;

virtual RCODE FLMAPI addSortKey(
	void *			pvSortKeyContext,
	FLMBOOL			bChildToContext,
	FLMBOOL			bElement,
	FLMUINT			uiNameId,
	FLMUINT			uiCompareRules,
	FLMUINT			uiLimit,
	FLMUINT			uiKeyComponent,
	FLMBOOL			bSortDescending,
	FLMBOOL			bSortMissingHigh,
	void **			ppvContext) = 0;
	
virtual RCODE FLMAPI enablePositioning( void) = 0;

virtual RCODE FLMAPI positionTo(
	IF_Db *			pDb,
	IF_DOMNode **	ppNode,
	FLMUINT			uiTimeLimit,
	FLMUINT			uiPosition) = 0;
	
virtual RCODE FLMAPI positionTo(
	IF_Db *				pDb,
	IF_DOMNode **		ppNode,
	FLMUINT				uiTimeLimit,
	IF_DataVector *	pSearchKey,
	FLMUINT				uiFlags) = 0;

virtual RCODE FLMAPI getPosition(
	IF_Db *				pDb,
	FLMUINT *			puiPosition) = 0;
	
virtual RCODE FLMAPI buildResultSet(
	IF_Db *	pDb,
	FLMUINT	uiTimeLimit) = 0;
	
virtual void FLMAPI stopBuildingResultSet( void) = 0;

virtual RCODE FLMAPI getCounts(
	IF_Db *		pDb,
	FLMUINT		uiTimeLimit,
	FLMBOOL		bPartialCountOk,
	FLMUINT *	puiReadCount,
	FLMUINT *	puiPassedCount,
	FLMUINT *	puiPositionableToCount,
	FLMBOOL *	pbDoneBuildingResultSet = NULL) = 0;
	
virtual void FLMAPI enableResultSetEncryption( void) = 0;

virtual void FLMAPI setQueryStatusObject(
	IF_QueryStatus *		pQueryStatus) = 0;

virtual void FLMAPI setQueryValidatorObject(
	IF_QueryValidator *		pQueryValidator) = 0;
*/

